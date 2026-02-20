import math


class _EmbeddingDbCompat:
    def add_embedding_dir(self, path):
        return None

    def load_textual_inversion_embeddings(self, force_reload=True, sync_with_sd_model=False):
        return None

    def register_embedding_by_name(self, embedding, sd_model, name):
        return None


class ModelHijackCompat:
    """
    Minimal compatibility shim for legacy extensions that import:
      from modules.sd_hijack import model_hijack
    """

    def __init__(self):
        self.embedding_db = _EmbeddingDbCompat()

    def get_prompt_lengths(self, prompt, cond_stage_model=None):
        # Legacy extensions may pass cond_stage_model; current core API does not require it.
        try:
            from modules import sd_models
            return sd_models.model_data.sd_model.get_prompt_lengths_on_ui(prompt)
        except Exception:
            # Fallback to a simple token estimate if model API is not ready yet.
            r = len(
                prompt.strip("!,. ")
                .replace(" ", ",")
                .replace(".", ",")
                .replace("!", ",")
                .replace(",,", ",")
                .replace(",,", ",")
                .replace(",,", ",")
                .replace(",,", ",")
                .split(",")
            )
            max_len = math.ceil(max(r, 1) / 75) * 75
            return r, max_len


model_hijack = ModelHijackCompat()
