import os
from collections import namedtuple

import torch

from backend.utils import load_torch_file
from modules import devices, errors, hashes, sd_models, shared  # noqa

TextualInversionTemplate = namedtuple("TextualInversionTemplate", ["name", "path"])
textual_inversion_templates = {}


def list_textual_inversion_templates():
    textual_inversion_templates.clear()

    for root, _, fns in os.walk(shared.cmd_opts.textual_inversion_templates_dir):
        for fn in fns:
            path = os.path.join(root, fn)

            textual_inversion_templates[fn] = TextualInversionTemplate(fn, path)

    return textual_inversion_templates


class Embedding:
    def __init__(self, vec, name, step=None):
        self.vec = vec
        self.name = name
        self.step = step
        self.shape = None
        self.vectors = 0
        self.cached_checksum = None
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None
        self.optimizer_state_dict = None
        self.filename = None
        self.hash = None
        self.shorthash = None

    def save(self, *args, **kwargs):
        raise NotImplementedError()

    def checksum(self):
        if self.cached_checksum is not None:
            return self.cached_checksum

        def const_hash(a):
            r = 0
            for v in a:
                r = (r * 281 ^ int(v) * 997) & 0xFFFFFFFF
            return r

        self.cached_checksum = f"{const_hash(self.vec.reshape(-1) * 100) & 0xffff:04x}"
        return self.cached_checksum

    def set_hash(self, v):
        self.hash = v
        self.shorthash = self.hash[0:12]


class DirWithTextualInversionEmbeddings:
    def __init__(self, path):
        self.path = path
        self.mtime = None

    def has_changed(self):
        if not os.path.isdir(self.path):
            return False

        mt = os.path.getmtime(self.path)
        if self.mtime is None or mt > self.mtime:
            return True

    def update(self):
        if not os.path.isdir(self.path):
            return

        self.mtime = os.path.getmtime(self.path)


class EmbeddingDatabase:
    def __init__(self):
        self.ids_lookup = {}
        self.word_embeddings = {}
        self.skipped_embeddings = {}
        self.expected_shape = -1
        self.embedding_dirs = {}
        self.previously_displayed_embeddings = ()

    def add_embedding_dir(self, path):
        self.embedding_dirs[path] = DirWithTextualInversionEmbeddings(path)

    def clear_embedding_dirs(self):
        self.embedding_dirs.clear()

    def register_embedding(self, embedding, model):
        return self.register_embedding_by_name(embedding, model, embedding.name)

    def register_embedding_by_name(self, embedding, model, name):
        ids = [0, 0, 0]  # model.cond_stage_model.tokenize([name])[0]
        first_id = ids[0]
        if first_id not in self.ids_lookup:
            self.ids_lookup[first_id] = []
        if name in self.word_embeddings:
            # remove old one from the lookup list
            lookup = [x for x in self.ids_lookup[first_id] if x[1].name != name]
        else:
            lookup = self.ids_lookup[first_id]
        if embedding is not None:
            lookup += [(ids, embedding)]
        self.ids_lookup[first_id] = sorted(lookup, key=lambda x: len(x[0]), reverse=True)
        if embedding is None:
            # unregister embedding with specified name
            if name in self.word_embeddings:
                del self.word_embeddings[name]
            if len(self.ids_lookup[first_id]) == 0:
                del self.ids_lookup[first_id]
            return None
        self.word_embeddings[name] = embedding
        return embedding

    def get_expected_shape(self):
        devices.torch_npu_set_device()
        vec = shared.sd_model.cond_stage_model.encode_embedding_init_text(",", 1)
        return vec.shape[1]

    def load_from_file(self, path, filename):
        name, ext = os.path.splitext(filename)
        if ext.lower() not in (".bin", ".pt", ".safetensors", ".sft"):
            return
        data = load_torch_file(path)
        if data is not None:
            embedding = create_embedding_from_data(data, name, filename=filename, filepath=path)
            self.register_embedding(embedding, None)
        else:
            print(f'Failed to load Textual Inversion "{name}"')

    def load_from_dir(self, embdir):
        if not os.path.isdir(embdir.path):
            return

        for root, _, fns in os.walk(embdir.path, followlinks=True):
            for fn in fns:
                try:
                    fullfn = os.path.join(root, fn)

                    if os.stat(fullfn).st_size == 0:
                        continue

                    self.load_from_file(fullfn, fn)
                except Exception:
                    errors.report(f"Error loading embedding {fn}", exc_info=True)
                    continue

    def load_textual_inversion_embeddings(self, force_reload=False, sync_with_sd_model=True):
        if not force_reload:
            need_reload = False
            for embdir in self.embedding_dirs.values():
                if embdir.has_changed():
                    need_reload = True
                    break

            if not need_reload:
                return

        self.ids_lookup.clear()
        self.word_embeddings.clear()
        self.skipped_embeddings.clear()

        if sync_with_sd_model:
            self.expected_shape = self.get_expected_shape()

        for embdir in self.embedding_dirs.values():
            self.load_from_dir(embdir)
            embdir.update()

        # re-sort word_embeddings because load_from_dir may not load in alphabetic order.
        # using a temporary copy so we don't reinitialize self.word_embeddings in case other objects have a reference to it.
        sorted_word_embeddings = {e.name: e for e in sorted(self.word_embeddings.values(), key=lambda e: e.name.lower())}
        self.word_embeddings.clear()
        self.word_embeddings.update(sorted_word_embeddings)

    def find_embedding_at_position(self, tokens, offset):
        token = tokens[offset]
        possible_matches = self.ids_lookup.get(token, None)

        if possible_matches is None:
            return None, None

        for ids, embedding in possible_matches:
            if tokens[offset : offset + len(ids)] == ids:
                return embedding, len(ids)

        return None, None


def create_embedding(name, num_vectors_per_token, overwrite_old, init_text="*"):
    cond_model = shared.sd_model.cond_stage_model

    with devices.autocast():
        cond_model([""])  # will send cond model to GPU if lowvram/medvram is active

    # cond_model expects at least some text, so we provide '*' as backup.
    embedded = cond_model.encode_embedding_init_text(init_text or "*", num_vectors_per_token)
    vec = torch.zeros((num_vectors_per_token, embedded.shape[1]), device=devices.device)

    # Only copy if we provided an init_text, otherwise keep vectors as zeros
    if init_text:
        for i in range(num_vectors_per_token):
            vec[i] = embedded[i * int(embedded.shape[0]) // num_vectors_per_token]

    # Remove illegal characters from name.
    name = "".join(x for x in name if (x.isalnum() or x in "._- "))
    fn = os.path.join(shared.cmd_opts.embeddings_dir, f"{name}.pt")
    if not overwrite_old:
        assert not os.path.exists(fn), f"file {fn} already exists"

    embedding = Embedding(vec, name)
    embedding.step = 0
    embedding.save(fn)

    return fn


def create_embedding_from_data(data, name, filename="unknown embedding file", filepath=None):
    if "string_to_param" in data:  # textual inversion embeddings
        param_dict = data["string_to_param"]
        param_dict = getattr(param_dict, "_parameters", param_dict)  # fix for torch 1.12.1 loading saved file from torch 1.11
        assert len(param_dict) == 1, "embedding file has multiple terms in it"
        emb = next(iter(param_dict.items()))[1]
        vec = emb.detach().to(devices.device, dtype=torch.float32)
        shape = vec.shape[-1]
        vectors = vec.shape[0]
    elif type(data) == dict and "clip_g" in data and "clip_l" in data:  # SDXL embedding
        vec = {k: v.detach().to(devices.device, dtype=torch.float32) for k, v in data.items()}
        shape = data["clip_g"].shape[-1] + data["clip_l"].shape[-1]
        vectors = data["clip_g"].shape[0]
    elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor:  # diffuser concepts
        assert len(data.keys()) == 1, "embedding file has multiple terms in it"

        emb = next(iter(data.values()))
        if len(emb.shape) == 1:
            emb = emb.unsqueeze(0)
        vec = emb.detach().to(devices.device, dtype=torch.float32)
        shape = vec.shape[-1]
        vectors = vec.shape[0]
    else:
        raise Exception(f"Couldn't identify {filename} as neither textual inversion embedding nor diffuser concept.")

    embedding = Embedding(vec, name)
    embedding.step = data.get("step", None)
    embedding.sd_checkpoint = data.get("sd_checkpoint", None)
    embedding.sd_checkpoint_name = data.get("sd_checkpoint_name", None)
    embedding.vectors = vectors
    embedding.shape = shape

    if filepath:
        embedding.filename = filepath
        embedding.set_hash(hashes.sha256(filepath, "textual_inversion/" + name) or "")

    return embedding
