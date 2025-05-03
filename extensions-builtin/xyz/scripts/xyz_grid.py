import random
from collections import namedtuple
from copy import copy
from itertools import permutations
from math import sqrt

import gradio as gr
import numpy as np
from lib_xyz import builtins
from lib_xyz.classes import AxisOption, SharedSettingsStackHelper
from lib_xyz.main import (
    draw_xyz_grid,
    re_range,
    re_range_count,
    re_range_count_float,
    re_range_float,
)
from lib_xyz.utils import csv_string_to_list_strip, list_to_csv_string, str_permutations
from PIL import Image

from modules import errors, images, scripts, shared
from modules.processing import (
    Processed,
    StableDiffusionProcessingTxt2Img,
    create_infotext,
    fix_seed,
    process_images,
)
from modules.shared import opts, state
from modules.ui_components import ToolButton

fill_values_symbol = "\U0001f4d2"  # 📒

AxisInfo = namedtuple("AxisInfo", ["axis", "values"])

axis_options = builtins.builtin_options


class XYZ(scripts.Script):
    def title(self):
        return "X/Y/Z Plot"

    def ui(self, is_img2img):
        self.current_axis_options = [x for x in axis_options if type(x) is AxisOption or x.is_img2img == is_img2img]

        with gr.Column():
            with gr.Row():
                x_type = gr.Dropdown(
                    label="X type",
                    choices=[x.label for x in self.current_axis_options],
                    value=self.current_axis_options[1].label,
                    type="index",
                    elem_id=self.elem_id("x_type"),
                    scale=1,
                )
                x_values = gr.Textbox(
                    label="X values",
                    lines=1,
                    elem_id=self.elem_id("x_values"),
                    scale=2,
                )
                x_values_dropdown = gr.Dropdown(
                    label="X values",
                    visible=False,
                    multiselect=True,
                    interactive=True,
                    scale=2,
                )
                fill_x_button = ToolButton(
                    value=fill_values_symbol,
                    elem_id="xyz_grid_fill_x_tool_button",
                    visible=False,
                    scale=1,
                )

            with gr.Row():
                y_type = gr.Dropdown(
                    label="Y type",
                    choices=[x.label for x in self.current_axis_options],
                    value=self.current_axis_options[0].label,
                    type="index",
                    elem_id=self.elem_id("y_type"),
                    scale=1,
                )
                y_values = gr.Textbox(
                    label="Y values",
                    lines=1,
                    elem_id=self.elem_id("y_values"),
                    scale=2,
                )
                y_values_dropdown = gr.Dropdown(
                    label="Y values",
                    visible=False,
                    multiselect=True,
                    interactive=True,
                    scale=2,
                )
                fill_y_button = ToolButton(
                    value=fill_values_symbol,
                    elem_id="xyz_grid_fill_y_tool_button",
                    visible=False,
                    scale=1,
                )

            with gr.Row():
                z_type = gr.Dropdown(
                    label="Z type",
                    choices=[x.label for x in self.current_axis_options],
                    value=self.current_axis_options[0].label,
                    type="index",
                    elem_id=self.elem_id("z_type"),
                    scale=1,
                )
                z_values = gr.Textbox(
                    label="Z values",
                    lines=1,
                    elem_id=self.elem_id("z_values"),
                    scale=2,
                )
                z_values_dropdown = gr.Dropdown(
                    label="Z values",
                    visible=False,
                    multiselect=True,
                    interactive=True,
                    scale=2,
                )
                fill_z_button = ToolButton(
                    value=fill_values_symbol,
                    elem_id="xyz_grid_fill_z_tool_button",
                    visible=False,
                    scale=1,
                )

        with gr.Group(elem_id="axis_options"):
            row_count = gr.Slider(
                label="Row Count",
                info="(set to 0 for auto)",
                minimum=0,
                maximum=8,
                value=0,
                step=1,
                elem_id=self.elem_id("row_count"),
            )
            with gr.Row(variant="compact"):
                with gr.Column(variant="compact", scale=3):
                    draw_legend = gr.Checkbox(
                        label="Draw legend",
                        value=True,
                        elem_id=self.elem_id("draw_legend"),
                    )
                    no_fixed_seeds = gr.Checkbox(
                        label="Keep -1 for seeds",
                        value=False,
                        elem_id=self.elem_id("no_fixed_seeds"),
                    )
                    with gr.Row(variant="compact"):
                        vary_seeds_x = gr.Checkbox(
                            label="Vary seeds for X",
                            value=False,
                            min_width=80,
                            elem_id=self.elem_id("vary_seeds_x"),
                            tooltip="Use different seeds for images along X axis.",
                        )
                        vary_seeds_y = gr.Checkbox(
                            label="Vary seeds for Y",
                            value=False,
                            min_width=80,
                            elem_id=self.elem_id("vary_seeds_y"),
                            tooltip="Use different seeds for images along Y axis.",
                        )
                        vary_seeds_z = gr.Checkbox(
                            label="Vary seeds for Z",
                            value=False,
                            min_width=80,
                            elem_id=self.elem_id("vary_seeds_z"),
                            tooltip="Use different seeds for images along Z axis.",
                        )
                with gr.Column(variant="compact", scale=2):
                    include_lone_images = gr.Checkbox(
                        label="Include Sub Images",
                        value=False,
                        elem_id=self.elem_id("include_lone_images"),
                    )
                    include_sub_grids = gr.Checkbox(
                        label="Include Sub Grids",
                        value=False,
                        elem_id=self.elem_id("include_sub_grids"),
                    )
                    csv_mode = gr.Checkbox(
                        label="Use text inputs instead of dropdowns",
                        value=False,
                        elem_id=self.elem_id("csv_mode"),
                    )
            margin_size = gr.Slider(
                label="Grid margins (px)",
                minimum=0,
                maximum=256,
                value=0,
                step=2,
                elem_id=self.elem_id("margin_size"),
            )

        with gr.Row(variant="compact", elem_id="swap_axes"):
            swap_xy_axes_button = gr.Button(value="Swap X/Y axes", elem_id="xy_grid_swap_axes_button")
            swap_yz_axes_button = gr.Button(value="Swap Y/Z axes", elem_id="yz_grid_swap_axes_button")
            swap_xz_axes_button = gr.Button(value="Swap X/Z axes", elem_id="xz_grid_swap_axes_button")

        def swap_axes(
            axis1_type,
            axis1_values,
            axis1_values_dropdown,
            axis2_type,
            axis2_values,
            axis2_values_dropdown,
        ):
            return (
                self.current_axis_options[axis2_type].label,
                axis2_values,
                axis2_values_dropdown,
                self.current_axis_options[axis1_type].label,
                axis1_values,
                axis1_values_dropdown,
            )

        xy_swap_args = [
            x_type,
            x_values,
            x_values_dropdown,
            y_type,
            y_values,
            y_values_dropdown,
        ]
        swap_xy_axes_button.click(swap_axes, inputs=xy_swap_args, outputs=xy_swap_args)

        yz_swap_args = [
            y_type,
            y_values,
            y_values_dropdown,
            z_type,
            z_values,
            z_values_dropdown,
        ]
        swap_yz_axes_button.click(swap_axes, inputs=yz_swap_args, outputs=yz_swap_args)

        xz_swap_args = [
            x_type,
            x_values,
            x_values_dropdown,
            z_type,
            z_values,
            z_values_dropdown,
        ]
        swap_xz_axes_button.click(swap_axes, inputs=xz_swap_args, outputs=xz_swap_args)

        def fill(axis_type, csv_mode):
            axis = self.current_axis_options[axis_type]
            if axis.choices:
                if csv_mode:
                    return list_to_csv_string(axis.choices()), gr.skip()
                else:
                    return gr.skip(), axis.choices()
            else:
                return gr.skip(), gr.skip()

        fill_x_button.click(fn=fill, inputs=[x_type, csv_mode], outputs=[x_values, x_values_dropdown])
        fill_y_button.click(fn=fill, inputs=[y_type, csv_mode], outputs=[y_values, y_values_dropdown])
        fill_z_button.click(fn=fill, inputs=[z_type, csv_mode], outputs=[z_values, z_values_dropdown])

        def select_axis(axis_type, axis_values, axis_values_dropdown, csv_mode):
            axis_type = axis_type or 0  # if axle type is None set to 0

            choices = self.current_axis_options[axis_type].choices
            has_choices = choices is not None

            if has_choices:
                choices = choices()
                if csv_mode:
                    if axis_values_dropdown:
                        axis_values = list_to_csv_string(list(filter(lambda x: x in choices, axis_values_dropdown)))
                        axis_values_dropdown = []
                else:
                    if axis_values:
                        axis_values_dropdown = list(
                            filter(
                                lambda x: x in choices,
                                csv_string_to_list_strip(axis_values),
                            )
                        )
                        axis_values = ""

            return (
                gr.Button.update(visible=has_choices),
                gr.Textbox.update(visible=((not has_choices) or csv_mode), value=axis_values),
                gr.update(
                    choices=choices if has_choices else None,
                    visible=has_choices and not csv_mode,
                    value=axis_values_dropdown,
                ),
            )

        x_type.change(
            fn=select_axis,
            inputs=[x_type, x_values, x_values_dropdown, csv_mode],
            outputs=[fill_x_button, x_values, x_values_dropdown],
        )
        y_type.change(
            fn=select_axis,
            inputs=[y_type, y_values, y_values_dropdown, csv_mode],
            outputs=[fill_y_button, y_values, y_values_dropdown],
        )
        z_type.change(
            fn=select_axis,
            inputs=[z_type, z_values, z_values_dropdown, csv_mode],
            outputs=[fill_z_button, z_values, z_values_dropdown],
        )

        def change_choice_mode(
            csv_mode,
            x_type,
            x_values,
            x_values_dropdown,
            y_type,
            y_values,
            y_values_dropdown,
            z_type,
            z_values,
            z_values_dropdown,
        ):
            _fill_x_button, _x_values, _x_values_dropdown = select_axis(x_type, x_values, x_values_dropdown, csv_mode)
            _fill_y_button, _y_values, _y_values_dropdown = select_axis(y_type, y_values, y_values_dropdown, csv_mode)
            _fill_z_button, _z_values, _z_values_dropdown = select_axis(z_type, z_values, z_values_dropdown, csv_mode)
            return (
                _fill_x_button,
                _x_values,
                _x_values_dropdown,
                _fill_y_button,
                _y_values,
                _y_values_dropdown,
                _fill_z_button,
                _z_values,
                _z_values_dropdown,
            )

        csv_mode.change(
            fn=change_choice_mode,
            inputs=[
                csv_mode,
                x_type,
                x_values,
                x_values_dropdown,
                y_type,
                y_values,
                y_values_dropdown,
                z_type,
                z_values,
                z_values_dropdown,
            ],
            outputs=[
                fill_x_button,
                x_values,
                x_values_dropdown,
                fill_y_button,
                y_values,
                y_values_dropdown,
                fill_z_button,
                z_values,
                z_values_dropdown,
            ],
        )

        def get_dropdown_update_from_params(axis, params):
            val_key = f"{axis} Values"
            vals = params.get(val_key, "")
            valslist = csv_string_to_list_strip(vals)
            return gr.update(value=valslist)

        self.infotext_fields = (
            (x_type, "X Type"),
            (x_values, "X Values"),
            (
                x_values_dropdown,
                lambda params: get_dropdown_update_from_params("X", params),
            ),
            (y_type, "Y Type"),
            (y_values, "Y Values"),
            (
                y_values_dropdown,
                lambda params: get_dropdown_update_from_params("Y", params),
            ),
            (z_type, "Z Type"),
            (z_values, "Z Values"),
            (
                z_values_dropdown,
                lambda params: get_dropdown_update_from_params("Z", params),
            ),
        )

        return [
            x_type,
            x_values,
            x_values_dropdown,
            y_type,
            y_values,
            y_values_dropdown,
            z_type,
            z_values,
            z_values_dropdown,
            draw_legend,
            include_lone_images,
            include_sub_grids,
            no_fixed_seeds,
            vary_seeds_x,
            vary_seeds_y,
            vary_seeds_z,
            row_count,
            margin_size,
            csv_mode,
        ]

    def run(
        self,
        p,
        x_type,
        x_values,
        x_values_dropdown,
        y_type,
        y_values,
        y_values_dropdown,
        z_type,
        z_values,
        z_values_dropdown,
        draw_legend,
        include_lone_images,
        include_sub_grids,
        no_fixed_seeds,
        vary_seeds_x,
        vary_seeds_y,
        vary_seeds_z,
        row_count,
        margin_size,
        csv_mode,
    ):
        x_type, y_type, z_type = (
            x_type or 0,
            y_type or 0,
            z_type or 0,
        )  # if axle type is None set to 0

        if not no_fixed_seeds:
            fix_seed(p)

        if not opts.return_grid:
            p.batch_size = 1

        def process_axis(opt, vals, vals_dropdown):
            if opt.label == "Nothing":
                return [0]

            if opt.choices is not None and not csv_mode:
                valslist = vals_dropdown
            elif opt.prepare is not None:
                valslist = opt.prepare(vals)
            else:
                valslist = csv_string_to_list_strip(vals)

            if opt.type is int:
                valslist_ext = []

                for val in valslist:
                    if val.strip() == "":
                        continue
                    m = re_range.fullmatch(val)
                    mc = re_range_count.fullmatch(val)
                    if m is not None:
                        start = int(m.group(1))
                        end = int(m.group(2)) + 1
                        step = int(m.group(3)) if m.group(3) is not None else 1

                        valslist_ext += list(range(start, end, step))
                    elif mc is not None:
                        start = int(mc.group(1))
                        end = int(mc.group(2))
                        num = int(mc.group(3)) if mc.group(3) is not None else 1

                        valslist_ext += [int(x) for x in np.linspace(start=start, stop=end, num=num).tolist()]
                    else:
                        valslist_ext.append(val)

                valslist = valslist_ext
            elif opt.type is float:
                valslist_ext = []

                for val in valslist:
                    if val.strip() == "":
                        continue
                    m = re_range_float.fullmatch(val)
                    mc = re_range_count_float.fullmatch(val)
                    if m is not None:
                        start = float(m.group(1))
                        end = float(m.group(2))
                        step = float(m.group(3)) if m.group(3) is not None else 1

                        valslist_ext += np.arange(start, end + step, step).tolist()
                    elif mc is not None:
                        start = float(mc.group(1))
                        end = float(mc.group(2))
                        num = int(mc.group(3)) if mc.group(3) is not None else 1

                        valslist_ext += np.linspace(start=start, stop=end, num=num).tolist()
                    else:
                        valslist_ext.append(val)

                valslist = valslist_ext
            elif opt.type == str_permutations:
                valslist = list(permutations(valslist))

            valslist = [opt.type(x) for x in valslist]

            # Confirm options are valid before starting
            if opt.confirm:
                opt.confirm(p, valslist)

            return valslist

        x_opt = self.current_axis_options[x_type]
        if x_opt.choices is not None and not csv_mode:
            x_values = list_to_csv_string(x_values_dropdown)
        xs = process_axis(x_opt, x_values, x_values_dropdown)

        y_opt = self.current_axis_options[y_type]
        if y_opt.choices is not None and not csv_mode:
            y_values = list_to_csv_string(y_values_dropdown)
        ys = process_axis(y_opt, y_values, y_values_dropdown)

        z_opt = self.current_axis_options[z_type]
        if z_opt.choices is not None and not csv_mode:
            z_values = list_to_csv_string(z_values_dropdown)
        zs = process_axis(z_opt, z_values, z_values_dropdown)

        # this could be moved to common code, but unlikely to be ever triggered anywhere else
        Image.MAX_IMAGE_PIXELS = None  # disable check in Pillow and rely on check below to allow large custom image sizes
        grid_mp = round(len(xs) * len(ys) * len(zs) * p.width * p.height / 1000000)
        assert grid_mp < opts.img_max_size_mp, f"Error: Resulting grid would be too large ({grid_mp} MPixels) (max configured size is {opts.img_max_size_mp} MPixels)"

        def fix_axis_seeds(axis_opt, axis_list):
            if axis_opt.label in ["Seed", "Var. seed"]:
                return [(int(random.randrange(4294967294)) if val is None or val == "" or val == -1 else val) for val in axis_list]
            else:
                return axis_list

        if not no_fixed_seeds:
            xs = fix_axis_seeds(x_opt, xs)
            ys = fix_axis_seeds(y_opt, ys)
            zs = fix_axis_seeds(z_opt, zs)

        if x_opt.label == "Steps":
            total_steps = sum(xs) * len(ys) * len(zs)
        elif y_opt.label == "Steps":
            total_steps = sum(ys) * len(xs) * len(zs)
        elif z_opt.label == "Steps":
            total_steps = sum(zs) * len(xs) * len(ys)
        else:
            total_steps = p.steps * len(xs) * len(ys) * len(zs)

        if isinstance(p, StableDiffusionProcessingTxt2Img) and p.enable_hr:
            if x_opt.label == "Hires steps":
                total_steps += sum(xs) * len(ys) * len(zs)
            elif y_opt.label == "Hires steps":
                total_steps += sum(ys) * len(xs) * len(zs)
            elif z_opt.label == "Hires steps":
                total_steps += sum(zs) * len(xs) * len(ys)
            elif p.hr_second_pass_steps:
                total_steps += p.hr_second_pass_steps * len(xs) * len(ys) * len(zs)
            else:
                total_steps *= 2

        total_steps *= p.n_iter

        image_cell_count = p.n_iter * p.batch_size
        _cell = f"; {image_cell_count} images per cell" if image_cell_count > 1 else ""
        _plural = "s" if len(zs) > 1 else ""
        _cnt = len(xs) * len(ys) * len(zs) * image_cell_count
        print(f"X/Y/Z plot will create {_cnt} images on {len(zs)} {len(xs)}x{len(ys)} grid{_plural}{_cell}")
        print(f"(Total steps: {total_steps})")
        shared.total_tqdm.updateTotal(total_steps)

        state.xyz_plot_x = AxisInfo(x_opt, xs)
        state.xyz_plot_y = AxisInfo(y_opt, ys)
        state.xyz_plot_z = AxisInfo(z_opt, zs)

        # If one of the axes is very slow to change between (like SD model
        # checkpoint), then make sure it is in the outer iteration of the nested
        # `for` loop.
        first_axes_processed = "z"
        second_axes_processed = "y"
        if x_opt.cost > y_opt.cost and x_opt.cost > z_opt.cost:
            first_axes_processed = "x"
            if y_opt.cost > z_opt.cost:
                second_axes_processed = "y"
            else:
                second_axes_processed = "z"
        elif y_opt.cost > x_opt.cost and y_opt.cost > z_opt.cost:
            first_axes_processed = "y"
            if x_opt.cost > z_opt.cost:
                second_axes_processed = "x"
            else:
                second_axes_processed = "z"
        elif z_opt.cost > x_opt.cost and z_opt.cost > y_opt.cost:
            first_axes_processed = "z"
            if x_opt.cost > y_opt.cost:
                second_axes_processed = "x"
            else:
                second_axes_processed = "y"

        grid_infotext = [None] * (1 + len(zs))

        def cell(x, y, z, ix, iy, iz):
            if shared.state.interrupted or state.stopping_generation:
                return Processed(p, [], p.seed, "")

            pc = copy(p)
            pc.styles = pc.styles[:]
            x_opt.apply(pc, x, xs)
            y_opt.apply(pc, y, ys)
            z_opt.apply(pc, z, zs)

            xdim = len(xs) if vary_seeds_x else 1
            ydim = len(ys) if vary_seeds_y else 1

            if vary_seeds_x:
                pc.seed += ix
            if vary_seeds_y:
                pc.seed += iy * xdim
            if vary_seeds_z:
                pc.seed += iz * xdim * ydim

            try:
                res = process_images(pc)
            except Exception as e:
                errors.display(e, "generating image for xyz plot")

                res = Processed(p, [], p.seed, "")

            # Sets subgrid infotexts
            subgrid_index = 1 + iz
            if grid_infotext[subgrid_index] is None and ix == 0 and iy == 0:
                pc.extra_generation_params = copy(pc.extra_generation_params)
                pc.extra_generation_params["Script"] = self.title()

                if x_opt.label != "Nothing":
                    pc.extra_generation_params["X Type"] = x_opt.label
                    pc.extra_generation_params["X Values"] = x_values
                    if x_opt.label in ["Seed", "Var. seed"] and not no_fixed_seeds:
                        pc.extra_generation_params["Fixed X Values"] = ", ".join([str(x) for x in xs])

                if y_opt.label != "Nothing":
                    pc.extra_generation_params["Y Type"] = y_opt.label
                    pc.extra_generation_params["Y Values"] = y_values
                    if y_opt.label in ["Seed", "Var. seed"] and not no_fixed_seeds:
                        pc.extra_generation_params["Fixed Y Values"] = ", ".join([str(y) for y in ys])

                grid_infotext[subgrid_index] = create_infotext(pc, pc.all_prompts, pc.all_seeds, pc.all_subseeds)

            # Sets main grid infotext
            if grid_infotext[0] is None and ix == 0 and iy == 0 and iz == 0:
                pc.extra_generation_params = copy(pc.extra_generation_params)

                if z_opt.label != "Nothing":
                    pc.extra_generation_params["Z Type"] = z_opt.label
                    pc.extra_generation_params["Z Values"] = z_values
                    if z_opt.label in ["Seed", "Var. seed"] and not no_fixed_seeds:
                        pc.extra_generation_params["Fixed Z Values"] = ", ".join([str(z) for z in zs])

                grid_infotext[0] = create_infotext(pc, pc.all_prompts, pc.all_seeds, pc.all_subseeds)

            return res

        with SharedSettingsStackHelper():
            processed = draw_xyz_grid(
                p,
                xs=xs,
                ys=ys,
                zs=zs,
                x_labels=[x_opt.format_value(p, x_opt, x) for x in xs],
                y_labels=[y_opt.format_value(p, y_opt, y) for y in ys],
                z_labels=[z_opt.format_value(p, z_opt, z) for z in zs],
                cell=cell,
                draw_legend=draw_legend,
                include_lone_images=include_lone_images,
                include_sub_grids=include_sub_grids,
                first_axes_processed=first_axes_processed,
                second_axes_processed=second_axes_processed,
                margin_size=margin_size,
            )

        if not processed.images:
            # It broke, no further handling needed.
            return processed

        z_count = len(zs)

        # Set the grid infotexts to the real ones with extra_generation_params (1 main grid + z_count sub-grids)
        processed.infotexts[: 1 + z_count] = grid_infotext[: 1 + z_count]

        if not include_lone_images:
            # Don't need sub-images anymore, drop from list:
            processed.images = processed.images[: z_count + 1]

        def rearrange_image(original_image: Image.Image, target_count: int):
            width, height = original_image.size
            width_per_image = int(width // len(xs))

            imgs_per_row = (len(xs) // target_count) + int(len(xs) % target_count != 0)
            print(f"Converting to {imgs_per_row}x{target_count} grid...")

            new_width = width_per_image * imgs_per_row
            new_height = height * target_count

            new_image = Image.new("RGB", (new_width, new_height), "white")

            for y in range(target_count):
                row = original_image.crop(
                    (
                        y * (width_per_image * imgs_per_row),
                        0,
                        min(width, ((y + 1) * (width_per_image * imgs_per_row))),
                        height,
                    )
                )
                new_image.paste(row, (0, height * y))

            return new_image

        if row_count == 0:  # Automatic
            if y_type + z_type > 0:
                target_count = 1
            else:
                target_count = int(sqrt(len(xs)))
                if target_count > 1:
                    print(f"Automatically split into {target_count} rows")
        else:
            if y_type + z_type == 0:
                target_count = row_count
            else:
                target_count = 1
                if row_count > 1:
                    print("\n[Error] Row Count currently only supports X-Axis...\n")

        if target_count > 1:
            processed.images[0] = rearrange_image(processed.images[0], target_count)

        if opts.grid_save:
            # Auto-save main and sub-grids:
            grid_count = z_count + 1 if z_count > 1 else 1
            for g in range(grid_count):
                # TODO: See previous comment about intentional data misalignment.
                adj_g = g - 1 if g > 0 else g
                images.save_image(
                    processed.images[g],
                    p.outpath_grids,
                    "xyz_grid",
                    info=processed.infotexts[g],
                    extension=opts.grid_format,
                    prompt=processed.all_prompts[adj_g],
                    seed=processed.all_seeds[adj_g],
                    grid=True,
                    p=processed,
                )
                if not include_sub_grids:  # if not include_sub_grids then skip saving after the first grid
                    break

        if not include_sub_grids:
            # Done with sub-grids, drop all related information:
            for _ in range(z_count):
                del processed.images[1]
                del processed.all_prompts[1]
                del processed.all_seeds[1]
                del processed.infotexts[1]

        return processed
