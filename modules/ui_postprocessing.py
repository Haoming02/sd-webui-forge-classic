import gradio as gr

import modules.infotext_utils as parameters_copypaste
from modules import call_queue, postprocessing, scripts, shared, ui_common, ui_toprow
from modules.ui_components import ResizeHandleRow


def create_ui():
    dummy_component = gr.Textbox(visible=False)
    tab_index = gr.State(value=0)

    with ResizeHandleRow(equal_height=False, variant="compact"):
        with gr.Column(variant="compact"):
            with gr.Tabs(elem_id="mode_extras"):
                with gr.Tab("Single Image", id="single_image", elem_id="extras_single_tab") as tab_single:
                    extras_image = gr.Image(label="Source", interactive=True, type="pil", elem_id="extras_image", image_mode="RGBA", height="48vh", sources="upload", show_download_button=False, show_share_button=False)

                with gr.Tab("Batch Process", id="batch_process", elem_id="extras_batch_process_tab") as tab_batch:
                    image_batch = gr.Files(label="Batch Process", interactive=True, elem_id="extras_image_batch")

                with gr.Tab("Batch from Directory", id="batch_from_directory", elem_id="extras_batch_directory_tab") as tab_batch_dir:
                    extras_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, placeholder="A directory on the same machine where the server is running.", elem_id="extras_batch_input_dir")
                    extras_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, placeholder="Leave blank to save images to the default path.", elem_id="extras_batch_output_dir")
                    show_extras_results = gr.Checkbox(label="Show result images", value=True, elem_id="extras_show_extras_results")

                with gr.Tab("Single Video", id="single_video", elem_id="extras_single_video") as tab_video:
                    extras_video_input = gr.Textbox(label="Input Clip", info='For video longer than 1 minute, please extract the frames and use "Batch from Directory" instead', **shared.hide_dirs, placeholder="Path to a video file", elem_id="extras_video_input")

            script_inputs = scripts.scripts_postproc.setup_ui()

        with gr.Column():
            toprow = ui_toprow.Toprow(is_compact=True, is_img2img=False, id_part="extras")
            toprow.create_inline_toprow_image()
            submit = toprow.submit

            output_panel = ui_common.create_output_panel("extras", shared.opts.outdir_extras_samples)

    tab_single.select(fn=lambda: 0, outputs=[tab_index], queue=False)
    tab_batch.select(fn=lambda: 1, outputs=[tab_index], queue=False)
    tab_batch_dir.select(fn=lambda: 2, outputs=[tab_index], queue=False)
    tab_video.select(fn=lambda: 3, outputs=[tab_index], queue=False)

    submit_click_inputs = [dummy_component, tab_index, extras_image, image_batch, extras_batch_input_dir, extras_batch_output_dir, show_extras_results, extras_video_input, *script_inputs]

    submit.click(
        fn=call_queue.wrap_gradio_gpu_call(postprocessing.run_postprocessing_webui, extra_outputs=[None, ""]),
        _js="submit_extras",
        inputs=submit_click_inputs,
        outputs=[
            output_panel.gallery,
            output_panel.generation_info,
            output_panel.html_log,
        ],
        show_progress=False,
    )

    parameters_copypaste.add_paste_fields("extras", extras_image, None)

    extras_image.change(fn=scripts.scripts_postproc.image_changed, queue=False)
