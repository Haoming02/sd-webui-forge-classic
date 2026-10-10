"""Wall-clock webui benchmark: real server + real checkpoint/LoRAs over the API.

Measures cold (first image incl. model load), warm per-image time, VRAM peaks
(torch reserved + NVML) and eviction events per config. Used to compare
offline vs dynamic vs wrapper LoRA paths and launch-flag profiles.

  python tests/dynamic_lora/bench_webui.py --runs '[["feat","offline","ck",1.0,"full"],["feat","dynamic","ck",1.0,"full"]]'
  python tests/dynamic_lora/bench_webui.py --quality        # rendered-image divergence check (dynamic vs wrapper vs offline)

Spec = [repo, mode(offline|dynamic|wrapper), attn(ck|sage), vram_fraction, profile]
Requires: webui worktrees for each repo key (default: this repo as "feat"),
checkpoints present per BENCH_CKPT / BENCH_OTHER / BENCH_LORAS env vars.
"""

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
REPOS = {"feat": str(REPO_ROOT), "neo": str(REPO_ROOT.parent / "bench-neo")}
DATA_DIR = REPO_ROOT  # worktrees have no models/; all servers share the main repo's data-dir
PYTHON = str(Path(sys.executable))

CKPT = os.environ.get("BENCH_CKPT", "gonzalomoKrea2_v40.safetensors")
OTHER = os.environ.get("BENCH_OTHER", "fluxKleinFP8_flux2Klein9bFp8.safetensors")
# benign LoRAs from models/Lora/krea2 (S drive): RealisticSnapshotKrea2 = standard r32 LoRA
# (low-rank addmm path), realism_engine_krea2_v2 = LoKr (kronecker einsum path).
LORAS = os.environ.get("BENCH_LORAS", "<lora:RealisticSnapshotKrea2:0.6> <lora:realism_engine_krea2_v2:0.4>")
SWAP = os.environ.get("BENCH_SWAP", "<lora:RealisticSnapshotKrea2:0.8> <lora:realism_engine_krea2_v2:0.2>")

COMMON = ["--sage", "--fast-fp16", "--fast-fp8"]
PROFILES = {
    "full": ["--enable-triton-backend", "--cuda-malloc", "--cuda-stream"],
    "min": [],
    "nostream": ["--enable-triton-backend", "--cuda-malloc", "--expandable-segments"],
    "notriton": ["--cuda-malloc", "--cuda-stream"],
    "expand": ["--enable-triton-backend", "--cuda-malloc", "--cuda-stream", "--expandable-segments"],
    "malloc_only": ["--cuda-malloc"],
}
PRESETS = {
    "core": [["feat", "offline", "ck", 1.0, "full"], ["feat", "dynamic", "ck", 1.0, "full"], ["neo", "wrapper", "ck", 1.0, "full"]],
    "constrained": [["feat", "dynamic", "ck", 0.5, "full"], ["neo", "wrapper", "ck", 0.5, "full"], ["neo", "offline", "ck", 0.5, "full"]],
    # triton x attention ablation: measured all-tied (6.1-6.3s steady, MAD<=0.1); expandable variant disproved the regression theory
    "attn4": [["feat", "dynamic", "ck", 1.0, "full"], ["feat", "dynamic", "ck", 1.0, "notriton"], ["feat", "dynamic", "sage", 1.0, "full"], ["feat", "dynamic", "sage", 1.0, "notriton"], ["feat", "dynamic", "ck", 1.0, "expand"]],
}
SCHED = os.environ.get("BENCH_SCHED", "<lora:RealisticSnapshotKrea2:[0.0@0.0,1.0@0.5,0.3@1.0]> <lora:realism_engine_krea2_v2:0.4>")


def nvidia_used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0])


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.stop_flag = threading.Event()

    def run(self):
        while not self.stop_flag.is_set():
            try:
                self.samples.append(nvidia_used())
            except Exception:
                pass
            time.sleep(0.25)

    def peak(self):
        return max(self.samples) if self.samples else 0


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def mad(xs):
    m = median(xs)
    return median([abs(x - m) for x in xs])


def launch(repo, tag, attn, profile, logdir):
    port = 7930 + (abs(hash(tag)) % 60)
    base = f"http://127.0.0.1:{port}"
    logdir = Path(logdir) if logdir else Path(tempfile.mkdtemp())
    logdir.mkdir(parents=True, exist_ok=True)
    logp = str(logdir / f"bench_{tag}.log")
    args = [PYTHON, "launch.py", "--api", "--nowebui", "--disable-gpu-warning", *COMMON, *PROFILES[profile], "--port", str(port), "--skip-torch-cuda-test", "--skip-version-check", "--skip-prepare-environment", "--data-dir", str(DATA_DIR)]
    if attn == "ck":
        args.append("--use-ck-attention")
    log = open(logp, "wb")
    proc = subprocess.Popen(args, cwd=REPOS[repo], stdout=log, stderr=subprocess.STDOUT)
    return proc, base, logp, log


def wait_ready(proc, base, timeout=420):
    deadline = time.time() + timeout
    while time.time() < deadline:
        assert proc.poll() is None, "server died"
        try:
            if requests.get(f"{base}/sdapi/v1/options", timeout=5).status_code == 200:
                return True
        except Exception:
            time.sleep(2)
    raise RuntimeError("server did not start")


def arm(base, mode, limit):
    requests.post(f"{base}/sdapi/v1/options", json={"setting_allocated_vram": limit}, timeout=30)
    online = "Automatic (fp16 LoRA)" if mode in ("wrapper", "dynamic") else "Automatic"
    requests.post(f"{base}/sdapi/v1/options", json={"sd_model_checkpoint": OTHER}, timeout=30)
    requests.post(f"{base}/sdapi/v1/options", json={"sd_model_checkpoint": CKPT, "forge_unet_storage_dtype": online}, timeout=30)


def gen(base, b):
    t0 = time.perf_counter()
    r = requests.post(f"{base}/sdapi/v1/txt2img", json=b, timeout=900)
    r.raise_for_status()
    return r.json(), round(time.perf_counter() - t0, 1)


def decode_image(resp_json):
    img = Image.open(io.BytesIO(base64.b64decode(resp_json["images"][0]))).convert("RGB")
    return np.asarray(img)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else round(10 * float(np.log10(255.0**2 / mse)), 1)


def pixel_diff(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    return {"mean_abs": round(float(d.mean()), 2), "frac_gt8": round(float((d > 8).mean()), 4)}


QUALITY_SEED = int(os.environ.get("BENCH_QSEED", "1234"))
# (repo, mode) arms; limit 1.0 to keep all arms fully resident -> cross-arm diffs isolate LoRA math only
QUALITY_ARMS = [("feat", "dynamic"), ("neo", "wrapper"), ("feat", "offline")]


def quality_one(repo, mode, outdir, attn="sage", profile="full", limit=1.0):
    proc = None
    log = None
    try:
        proc, base, logp, log = launch(repo, f"q_{mode}", attn, profile, outdir)
        wait_ready(proc, base)
        arm(base, mode, limit)
        body = {"prompt": f"1girl, photo {LORAS}", "negative_prompt": "", "cfg_scale": 1.0, "width": 1024, "height": 1024, "batch_size": 1, "sampler_name": "Euler", "enable_hr": False, "seed": QUALITY_SEED}
        b1 = dict(body, steps=1)
        b8 = dict(body, steps=8)
        gen(base, b8)  # untimed warmup: load + JIT
        a = decode_image(gen(base, b1)[0])  # single-application equivalence
        b = decode_image(gen(base, b1)[0])  # determinism floor
        img8 = decode_image(gen(base, b8)[0])
        Image.fromarray(a).save(f"{outdir}/quality1_{mode}.png")
        Image.fromarray(img8).save(f"{outdir}/quality8_{mode}.png")
        return {"mode": mode, "img_a": a, "img_b": b, "img8": img8, "log": logp}
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except Exception:
                proc.kill()
        if log is not None:
            log.close()


def quality(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    runs = {m: quality_one(r, m, outdir) for r, m in QUALITY_ARMS}
    report = {"seed": QUALITY_SEED, "prompt_loras": LORAS}
    report["determinism_floor_psnr"] = {m: psnr(v["img_a"], v["img_b"]) for m, v in runs.items()}
    pairs = [("dynamic", "wrapper"), ("dynamic", "offline"), ("wrapper", "offline")]
    report["steps1_psnr"] = {f"{x}_vs_{y}": psnr(runs[x]["img_a"], runs[y]["img_a"]) for x, y in pairs}
    report["steps1_pixdiff"] = {f"{x}_vs_{y}": pixel_diff(runs[x]["img_a"], runs[y]["img_a"]) for x, y in pairs}
    # 8-step trajectories are chaotic across ANY two numeric paths (upstream wrapper vs offline
    # differ the same way); reported for the record, and sanity-checked not-garbage only.
    report["steps8_psnr"] = {f"{x}_vs_{y}": psnr(runs[x]["img8"], runs[y]["img8"]) for x, y in pairs}
    s1 = report["steps1_psnr"]
    gates = {
        "single_application_dyn_vs_wrap>=32": s1["dynamic_vs_wrapper"] >= 32,
        "dyn_closer_to_wrap_than_offline": s1["dynamic_vs_wrapper"] >= s1["dynamic_vs_offline"],
        "determinism_floor>=40": all(v >= 40 for v in report["determinism_floor_psnr"].values()),
        "steps8_not_garbage": all(v >= 12 for v in report["steps8_psnr"].values()),
    }
    report["gates"] = gates
    print(json.dumps(report, indent=1), flush=True)
    if not all(gates.values()):
        raise SystemExit(1)


def one(tag, repo, mode, attn, limit, profile, reps=7, logdir=None):
    proc = log = None
    try:
        proc, base, logp, log = launch(repo, tag, attn, profile, logdir)
        wait_ready(proc, base)
        arm(base, mode, limit)

        body = {"prompt": "1girl, photo", "steps": 4, "cfg_scale": 1.0, "width": 1024, "height": 1024, "sampler_name": "Euler", "seed": 1234}
        body_lora = dict(body, prompt=f"1girl, photo {LORAS}")
        body_swap = dict(body_lora, prompt=f"1girl, photo {SWAP}")

        gen(base, body)  # untimed: model load + TE/VAE/allocator JIT priming

        # cold: forced model reload, first image WITHOUT loras (pure load/quant latency)
        requests.post(f"{base}/sdapi/v1/options", json={"sd_model_checkpoint": OTHER}, timeout=30)
        requests.post(f"{base}/sdapi/v1/options", json={"sd_model_checkpoint": CKPT}, timeout=30)
        _, t_cold = gen(base, body)

        # lora apply + strength swap: THE differentiator (offline requantizes; dynamic edits slices)
        _, t_lora_apply = gen(base, body_lora)
        _, t_lora_1 = gen(base, body_lora)
        _, t_swap = gen(base, body_swap)

        s = Sampler()
        s.start()
        steps1 = [gen(base, dict(body_lora, steps=1, seed=555))[1] for _ in range(4)]
        full = [gen(base, body_lora)[1] for _ in range(reps)]
        s.stop_flag.set()
        s.join(timeout=5)

        t_sched = t_sched_warm = None
        if mode in ("wrapper", "dynamic"):  # lora-ctl scheduled strengths (w@t syntax)
            _, t_sched = gen(base, dict(body, prompt=f"1girl, photo {SCHED}"))
            _, t_sched_warm = gen(base, dict(body, prompt=f"1girl, photo {SCHED}"))

        mem = requests.get(f"{base}/sdapi/v1/memory", timeout=30).json()["cuda"]
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except Exception:
                proc.kill()
        if log is not None:
            log.close()

    text = open(logp, errors="replace").read()
    return {
        "tag": tag,
        "repo": repo,
        "mode": mode,
        "attn": attn,
        "limit": limit,
        "profile": profile,
        "cold": t_cold,
        "lora_apply": t_lora_apply,
        "lora_apply_settled": t_lora_1,
        "strength_swap": t_swap,
        "steps1": steps1,
        "warm_all": full,
        "warm_median": median(full),
        "warm_mad": mad(full),
        "warm_min": min(full),
        "sched": t_sched,
        "sched_warm": t_sched_warm,
        "torch_reserved_peak_MB": round(mem["reserved"]["peak"] / 1e6),
        "nvml_peak_MB": s.peak(),
        "oom": mem["events"]["oom"],
        "retries": mem["events"]["retries"],
        "on_the_fly": ("on_the_fly = True" in text) if mode != "offline" else ("on_the_fly = False" in text),
        "loaded_partial_events": text.count("loaded partially"),
        "scheduler_errors": text.count("Invalid Syntax") + text.count("requires on-the-fly"),
        "log": logp,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", help="JSON list of [repo,mode,attn,limit,profile] (or with leading tag)")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="named run set (core | constrained | attn4)")
    ap.add_argument("--quality", action="store_true", help="render-mode divergence check (dynamic vs wrapper vs offline)")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--logdir", default=None)
    args = ap.parse_args()
    if args.quality:
        quality(args.logdir or tempfile.mkdtemp())
        raise SystemExit(0)
    specs = PRESETS[args.preset] if args.preset else json.loads(args.runs)
    for spec in specs:
        spec = list(spec)
        if len(spec) == 5:
            spec = [f"{spec[0]}_{spec[1]}_{spec[2]}_{spec[3]}_{spec[4]}", *spec]
        print(json.dumps(one(*spec, reps=args.reps, logdir=args.logdir)), flush=True)
