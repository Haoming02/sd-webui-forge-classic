import torch


class Emphasis:
    """Emphasis class determines how to deal with (emphasized texts:1.1) in prompts"""

    name: str = ""
    description: str = ""

    def __call__(self, z: torch.Tensor, multipliers: torch.Tensor):

        pass


class EmphasisNone(Emphasis):
    name = "None"
    description = "disable Emphasis and treat parentheses as literal characters"


class EmphasisIgnore(Emphasis):
    name = "Ignore"
    description = "consume parentheses but ignore all emphasis"


class EmphasisOriginal(Emphasis):
    name = "Original"
    description = "the default emphasis implementation"

    def __call__(self, z: torch.Tensor, multipliers: torch.Tensor):
        orig_mean = z.mean()
        z *= multipliers.reshape(multipliers.shape + (1,)).expand(z.shape)
        new_mean = z.mean()
        z *= orig_mean / new_mean
        return z


class EmphasisOriginalNoNorm(EmphasisOriginal):
    name = "No norm"
    description = "implementation without normalization (for SD1/SDXL only)"

    def __call__(self, z: torch.Tensor, multipliers: torch.Tensor):
        return z * multipliers.reshape(multipliers.shape + (1,)).expand(z.shape)


options: list[Emphasis] = [
    EmphasisNone,
    EmphasisIgnore,
    EmphasisOriginal,
    EmphasisOriginalNoNorm,
]


def get_current_option(emphasis_option_name: str) -> Emphasis:
    return next(iter([x for x in options if x.name == emphasis_option_name]), EmphasisOriginal)


def get_options_descriptions() -> str:
    return f"""
        <ul style='margin-left: 1.5em'><li>
            {"</li><li>".join(f"<b>{x.name}</b>: {x.description}" for x in options)}
        </li></ul>
            """


# region Utils


from .parsing import parse_prompt_attention


def uses_emphasis(prompt: str) -> bool:
    attention = parse_prompt_attention(prompt)
    return any(w != 1.0 for (_, w) in attention)
