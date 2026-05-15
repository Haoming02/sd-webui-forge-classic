import torch


class Emphasis:
    """Emphasis class decides how to death with (emphasized:1.1) text in prompts"""

    name: str = "Base"
    description: str = ""

    tokens: list[list[int]]
    """tokens from the chunk of the prompt"""

    multipliers: torch.Tensor
    """tensor with multipliers, once for each token"""

    z: torch.Tensor
    """output of cond transformers network (CLIP)"""

    def after_transformers(self):
        """Called after cond transformers network has processed the chunk of the prompt; this function should modify self.z to apply the emphasis"""

        pass


class EmphasisNone(Emphasis):
    name = "None"
    description = "disable Emphasis entirely and treat (:1.2) as literal characters"


class EmphasisIgnore(Emphasis):
    name = "Ignore"
    description = "treat all words as if they have no emphasis"


class EmphasisOriginal(Emphasis):
    name = "Original"
    description = "the original emphasis implementation"

    def after_transformers(self):
        original_mean = self.z.mean()
        self.z = self.z * self.multipliers.reshape(self.multipliers.shape + (1,)).expand(self.z.shape)
        new_mean = self.z.mean()
        self.z = self.z * (original_mean / new_mean)


class EmphasisOriginalNoNorm(EmphasisOriginal):
    name = "No norm"
    description = "implementation without normalization (fix certain issues for SDXL)"

    def after_transformers(self):
        self.z = self.z * self.multipliers.reshape(self.multipliers.shape + (1,)).expand(self.z.shape)


def get_current_option(emphasis_option_name):
    return next(iter([x for x in options if x.name == emphasis_option_name]), EmphasisOriginal)


def get_options_descriptions():
    return f"""
        <ul style='margin-left: 1.5em'><li>
            {"</li><li>".join(f"<b>{x.name}</b>: {x.description}" for x in options)}
        </li></ul>
            """


options = [
    EmphasisNone,
    EmphasisIgnore,
    EmphasisOriginal,
    EmphasisOriginalNoNorm,
]


# region Utils


from modules.prompt_parser import parse_prompt_attention


def uses_emphasis(prompt: str) -> bool:
    attention = parse_prompt_attention(prompt)
    return len(attention) != len([p for p in attention if p[1] == 1.0 or p[0] == "BREAK"])
