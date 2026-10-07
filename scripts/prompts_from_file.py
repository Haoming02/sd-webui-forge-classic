import copy
import shlex

import gradio as gr

import modules.scripts as scripts
from modules import errors, sd_models, sd_samplers
from modules.processing import Processed, fix_seed, process_images
from modules.shared import state


def process_model_tag(tag):
    info = sd_models.get_closet_checkpoint_match(tag)
    assert info is not None, f"Unknown checkpoint: {tag}"
    return info.name


def process_string_tag(tag):
    return tag


def process_int_tag(tag):
    return int(tag)


def process_float_tag(tag):
    return float(tag)


def process_boolean_tag(tag):
    return True if (tag == "true") else False


prompt_tags = {
    "sd_model": process_model_tag,
    "outpath_samples": process_string_tag,
    "outpath_grids": process_string_tag,
    "prompt_for_display": process_string_tag,
    "prompt": process_string_tag,
    "negative_prompt": process_string_tag,
    "styles": process_string_tag,
    "seed": process_int_tag,
    "subseed_strength": process_float_tag,
    "subseed": process_int_tag,
    "seed_resize_from_h": process_int_tag,
    "seed_resize_from_w": process_int_tag,
    "sampler_index": process_int_tag,
    "sampler_name": process_string_tag,
    "batch_size": process_int_tag,
    "n_iter": process_int_tag,
    "steps": process_int_tag,
    "cfg_scale": process_float_tag,
    "width": process_int_tag,
    "height": process_int_tag,
    "restore_faces": process_boolean_tag,
    "tiling": process_boolean_tag,
    "do_not_save_samples": process_boolean_tag,
    "do_not_save_grid": process_boolean_tag,
}


def cmdargs(line):
    args = shlex.split(line)
    pos = 0
    res = {}

    while pos < len(args):
        arg = args[pos]

        assert arg.startswith("--"), f'must start with "--": {arg}'
        assert pos + 1 < len(args), f"missing argument for command line option {arg}"

        tag = arg[2:]

        if tag == "prompt" or tag == "negative_prompt":
            pos += 1
            prompt = args[pos]
            pos += 1
            while pos < len(args) and not args[pos].startswith("--"):
                prompt += " "
                prompt += args[pos]
                pos += 1
            res[tag] = prompt
            continue

        func = prompt_tags.get(tag, None)
        assert func, f"unknown commandline option: {arg}"

        val = args[pos + 1]
        if tag == "sampler_name":
            val = sd_samplers.samplers_map.get(val.lower(), None)

        res[tag] = func(val)

        pos += 2

    return res


def load_prompt_file(file):
    if file is None:
        return None, gr.skip()
    else:
        lines = [x.strip() for x in file.decode("utf8", errors="ignore").split("\n")]
        return None, gr.update(value="\n".join(lines), lines=7)


class PromptsFromTexts(scripts.Script):
    def title(self):
        return "Prompts from File or Textbox"

    def ui(self, is_img2img):
        checkbox_iterate = gr.Checkbox(value=False, label="Iterate seed every line", elem_id=self.elem_id("checkbox_iterate"))
        checkbox_iterate_batch = gr.Checkbox(value=False, label="Use same random seed for all lines", elem_id=self.elem_id("checkbox_iterate_batch"))
        prompt_position = gr.Radio(label="Insert prompts at the", choices=("start", "end"), value="start", elem_id=self.elem_id("prompt_position"))

        prompt_txt = gr.Textbox(label="List of prompt inputs", lines=2, elem_id=self.elem_id("prompt_txt"))
        file = gr.File(label="Upload prompt inputs", type="binary", elem_id=self.elem_id("file"))

        prompt_txt.change(lambda tb: gr.update(lines=7) if ("\n" in tb) else gr.update(lines=2), inputs=[prompt_txt], outputs=[prompt_txt], show_progress=False)
        file.change(fn=load_prompt_file, inputs=[file], outputs=[file, prompt_txt], show_progress=False)

        return [checkbox_iterate, checkbox_iterate_batch, prompt_position, prompt_txt]

    def run(self, p, checkbox_iterate: bool, checkbox_iterate_batch: bool, prompt_position: str, prompt_txt: str):
        lines = [x for x in (x.strip() for x in prompt_txt.splitlines()) if x]

        p.do_not_save_grid = True

        job_count = 0
        jobs = []

        for line in lines:
            if "--" in line:
                try:
                    args = cmdargs(line)
                except Exception:
                    errors.report(f'Error parsing line "{line}"', exc_info=True)
                    args = {"prompt": line}
            else:
                args = {"prompt": line}

            job_count += args.get("n_iter", p.n_iter)

            jobs.append(args)

        print(f"Processing {len(lines)} lines in {job_count} jobs")
        if (checkbox_iterate or checkbox_iterate_batch) and p.seed == -1:
            fix_seed(p)

        state.job_count = job_count

        images = []
        all_prompts = []
        infotexts = []
        for args in jobs:
            state.job = f"{state.job_no + 1} out of {state.job_count}"

            copy_p = copy.copy(p)
            for k, v in args.items():
                if k == "sd_model":
                    copy_p.override_settings["sd_model_checkpoint"] = v
                else:
                    setattr(copy_p, k, v)

            if args.get("prompt") and p.prompt:
                if prompt_position == "start":
                    copy_p.prompt = f'{args.get("prompt")} {p.prompt}'
                else:
                    copy_p.prompt = f'{p.prompt} {args.get("prompt")}'

            if args.get("negative_prompt") and p.negative_prompt:
                if prompt_position == "start":
                    copy_p.negative_prompt = f'{args.get("negative_prompt")} {p.negative_prompt}'
                else:
                    copy_p.negative_prompt = f'{p.negative_prompt} {args.get("negative_prompt")}'

            proc = process_images(copy_p)
            images += proc.images

            if checkbox_iterate:
                p.seed = p.seed + (p.batch_size * p.n_iter)

            all_prompts += proc.all_prompts
            infotexts += proc.infotexts

        return Processed(p, images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)
