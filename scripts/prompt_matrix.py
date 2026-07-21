import logging
import math

import gradio as gr

import modules.scripts as scripts
from backend.logging import setup_logger
from modules import images
from modules.processing import fix_seed, process_images
from modules.shared import opts, state

logger = logging.getLogger("PromptMatrix")
setup_logger(logger)


def draw_xy_grid(xs, ys, x_label, y_label, cell):
    res = []

    ver_texts = [[images.GridAnnotation(y_label(y))] for y in ys]
    hor_texts = [[images.GridAnnotation(x_label(x))] for x in xs]

    first_processed = None

    state.job_count = len(xs) * len(ys)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            state.job = f"{ix + iy * len(xs) + 1} out of {len(xs) * len(ys)}"

            processed = cell(x, y)
            if first_processed is None:
                first_processed = processed

            res.append(processed.images[0])

    grid = images.image_grid(res, rows=len(ys))
    grid = images.draw_grid_annotations(grid, res[0].width, res[0].height, hor_texts, ver_texts)

    first_processed.images = [grid]
    return first_processed


class PromptMatrix(scripts.Script):
    def title(self):
        return "Prompt Matrix"

    def ui(self, is_img2img):
        gr.HTML("<br>")
        with gr.Row():
            with gr.Column():
                put_at_start = gr.Checkbox(value=False, label="Put the variable parts at the start of prompt", elem_id=self.elem_id("put_at_start"))
                different_seeds = gr.Checkbox(value=False, label="Use different seeds for each image", elem_id=self.elem_id("different_seeds"))
                margin_size = gr.Slider(value=0, label="Grid Margins (px)", minimum=0, maximum=256, step=2, elem_id=self.elem_id("margin_size"))
            with gr.Column():
                prompt_type = gr.Radio(value="positive", label="Prompt", choices=("positive", "negative"), elem_id=self.elem_id("prompt_type"))
                variations_delimiter = gr.Radio(value="comma", label="Joining Char.", choices=("comma", "space"), elem_id=self.elem_id("variations_delimiter"))

        return [put_at_start, different_seeds, prompt_type, variations_delimiter, margin_size]

    def run(self, p, put_at_start: bool, different_seeds: bool, prompt_type: str, variations_delimiter: str, margin_size: int):
        fix_seed(p)

        assert prompt_type in ("positive", "negative")
        assert variations_delimiter in ("comma", "space")

        prompt = p.prompt if prompt_type == "positive" else p.negative_prompt
        original_prompt = prompt[0] if isinstance(prompt, list) else prompt
        positive_prompt = p.prompt[0] if isinstance(p.prompt, list) else p.prompt

        delimiter = ", " if variations_delimiter == "comma" else " "

        all_prompts = []
        prompt_matrix_parts = original_prompt.split("|")
        combination_count = 2 ** (len(prompt_matrix_parts) - 1)
        for combination_num in range(combination_count):
            selected_prompts = [text.strip().strip(",") for n, text in enumerate(prompt_matrix_parts[1:]) if combination_num & (1 << n)]

            if put_at_start:
                selected_prompts = selected_prompts + [prompt_matrix_parts[0]]
            else:
                selected_prompts = [prompt_matrix_parts[0]] + selected_prompts

            all_prompts.append(delimiter.join(selected_prompts))

        if p.batch_size > 1:
            logger.warning("Enforcing Batch Size of 1")
            p.batch_size = 1

        p.n_iter = math.ceil(len(all_prompts) / p.batch_size)
        p.do_not_save_grid = True

        logger.info(f"Creating {len(all_prompts)} images in {p.n_iter} batches")

        if prompt_type == "positive":
            p.prompt = all_prompts
        else:
            p.negative_prompt = all_prompts

        p.prompt_for_display = positive_prompt
        p.seed = [p.seed + (i if different_seeds else 0) for i in range(len(all_prompts))]

        processed = process_images(p)

        grid = images.image_grid(processed.images, p.batch_size, rows=1 << ((len(prompt_matrix_parts) - 1) // 2))
        grid = images.draw_prompt_matrix(grid, processed.images[0].width, processed.images[0].height, prompt_matrix_parts, margin_size)

        processed.images.insert(0, grid)
        processed.index_of_first_image = 1
        processed.infotexts.insert(0, processed.infotexts[0])

        if opts.grid_save:
            images.save_image(processed.images[0], p.outpath_grids, "prompt_matrix", extension=opts.grid_format, prompt=original_prompt, seed=processed.seed, grid=True, p=p)

        return processed
