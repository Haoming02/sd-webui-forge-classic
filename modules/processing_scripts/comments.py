import re
from typing import TYPE_CHECKING

from modules import scripts, shared

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing


def strip_comments(text: str) -> str:
    if not shared.opts.enable_prompt_comments:
        return text

    # multi line comment (/* */)
    text = re.sub(r"\/\*.*?\*\/", "", text, flags=re.DOTALL)
    # single line comment (# | //)
    text = re.sub(r"[^\S\n]*(\#|\/\/).*", "", text)

    return text


class ScriptComments(scripts.Script):
    def title(self):
        return "Comments"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process(self, p: "StableDiffusionProcessing", *args):
        if not shared.opts.enable_prompt_comments:
            return

        if shared.opts.save_prompt_comments:
            p._all_prompts_c = p.all_prompts.copy()
            p._all_negative_prompts_c = p.all_negative_prompts.copy()

        p.all_prompts = [strip_comments(x) for x in p.all_prompts]
        p.all_negative_prompts = [strip_comments(x) for x in p.all_negative_prompts]

        p.main_prompt = strip_comments(p.main_prompt)
        p.main_negative_prompt = strip_comments(p.main_negative_prompt)

        if getattr(p, "enable_hr", False):
            p.all_hr_prompts = [strip_comments(x) for x in p.all_hr_prompts]
            p.all_hr_negative_prompts = [strip_comments(x) for x in p.all_hr_negative_prompts]

            p.hr_prompt = strip_comments(p.hr_prompt)
            p.hr_negative_prompt = strip_comments(p.hr_negative_prompt)


shared.options_templates.update(
    shared.options_section(
        ("ui_comments", "Comments", "ui"),
        {
            "enable_prompt_comments": shared.OptionInfo(True, "Remove Comments from Prompts").html(
                """
<b>Comment Syntax:</b><br>
<ul style='margin-left: 2em'>
<li># ...</li>
<li>// ...</li>
<li>/* ... */</li>
</ul>
                """
            ),
            "save_prompt_comments": shared.OptionInfo(False, "Save Raw Comments").info("include the comments in Infotext"),
        },
    )
)
