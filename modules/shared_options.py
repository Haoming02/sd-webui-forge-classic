import os
import gradio as gr

from modules import localization, ui_components, shared_items, shared, shared_gradio_themes, util, sd_emphasis
from modules.paths_internal import data_path, default_output_dir
from modules.shared_cmd_options import cmd_opts
from modules.options import options_section, OptionInfo, OptionHTML, categories

options_templates = {}
hide_dirs = shared.hide_dirs

restricted_opts = {
    "samples_filename_pattern",
    "directories_filename_pattern",
    "outdir_samples",
    "outdir_txt2img_samples",
    "outdir_img2img_samples",
    "outdir_extras_samples",
    "outdir_grids",
    "outdir_txt2img_grids",
    "outdir_save",
    "outdir_init_images",
}

categories.register_category("saving", "Saving images")
categories.register_category("sd", "Stable Diffusion")
categories.register_category("ui", "User Interface")
categories.register_category("system", "System")
categories.register_category("postprocessing", "Postprocessing")
categories.register_category("training", "Training")

options_templates.update(
    options_section(
        ("saving-images", "Saving images/grids", "saving"),
        {
            "samples_save": OptionInfo(True, "Always save all generated images"),
            "samples_format": OptionInfo("png", "File format for images", gr.Dropdown, {"choices": ("jpg", "jpeg", "png", "webp", "avif", "heif")}),
            "samples_filename_pattern": OptionInfo("", "Images filename pattern", component_args=hide_dirs).link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Images-Filename-Name-and-Subdirectory"),
            "save_images_add_number": OptionInfo(True, "Add number to filename when saving", component_args=hide_dirs),
            "save_images_replace_action": OptionInfo("Replace", "Saving the image to an existing file", gr.Radio, {"choices": ["Replace", "Add number suffix"], **hide_dirs}),
            "grid_save": OptionInfo(True, "Always save all generated image grids"),
            "grid_format": OptionInfo("png", "File format for grids", gr.Dropdown, {"choices": ("jpg", "jpeg", "png", "webp", "avif", "heif")}),
            "grid_extended_filename": OptionInfo(False, "Add extended info (seed, prompt) to filename when saving grid"),
            "grid_only_if_multiple": OptionInfo(True, "Do not save grids consisting of one picture"),
            "grid_prevent_empty_spots": OptionInfo(False, "Prevent empty spots in grid (when set to autodetect)"),
            "grid_zip_filename_pattern": OptionInfo("", "Archive filename pattern", component_args=hide_dirs).link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Images-Filename-Name-and-Subdirectory"),
            "n_rows": OptionInfo(-1, "Grid row count; use -1 for autodetect and 0 for it to be same as batch size", gr.Slider, {"minimum": -1, "maximum": 16, "step": 1}),
            "font": OptionInfo("", "Font for image grids that have text"),
            "grid_text_active_color": OptionInfo("#000000", "Text color for image grids", ui_components.FormColorPicker, {}),
            "grid_text_inactive_color": OptionInfo("#999999", "Inactive text color for image grids", ui_components.FormColorPicker, {}),
            "grid_background_color": OptionInfo("#ffffff", "Background color for image grids", ui_components.FormColorPicker, {}),
            "save_images_before_face_restoration": OptionInfo(False, "Save a copy of image before doing face restoration."),
            "save_images_before_highres_fix": OptionInfo(False, "Save a copy of image before applying highres fix."),
            "save_images_before_color_correction": OptionInfo(False, "Save a copy of image before applying color correction to img2img results"),
            "save_mask": OptionInfo(False, "For inpainting, save a copy of the greyscale mask"),
            "save_mask_composite": OptionInfo(False, "For inpainting, save a masked composite"),
            "jpeg_quality": OptionInfo(80, "Quality for saved jpeg images", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
            "webp_lossless": OptionInfo(False, "Use lossless compression for webp images"),
            "opt_image": OptionInfo(False, "Enable optimization when saving images").info("slightly reduce file size; take longer to process"),
            "export_for_4chan": OptionInfo(True, "Save copy of large images as JPG").info("if the file size is above the limit, or either width or height are above the limit"),
            "img_downscale_threshold": OptionInfo(4.0, "File size limit for the above option", gr.Number).info("in MB"),
            "target_side_length": OptionInfo(4096, "Width/height limit for the above option", gr.Number).info("in pixels"),
            "img_max_size_mp": OptionInfo(100, "Maximum image size", gr.Number).info("in megapixels; only affect <b>X/Y/Z Plot</b>"),
            "use_original_name_batch": OptionInfo(True, "Use original name for output filename during batch process in extras tab"),
            "use_upscaler_name_as_suffix": OptionInfo(False, "Use upscaler name as filename suffix in the extras tab"),
            "save_selected_only": OptionInfo(True, "When using 'Save' button, only save a single selected image"),
            "save_init_img": OptionInfo(False, "Save init images when using img2img"),
            "temp_dir": OptionInfo("", "Directory for temporary images; leave empty for default"),
            "clean_temp_dir_at_start": OptionInfo(False, "Cleanup non-default temporary directory when starting webui"),
            "save_incomplete_images": OptionInfo(False, "Save incomplete images").info("save images that has been interrupted in mid-generation; even if not saved, they will still show up in webui output."),
            "notification_audio": OptionInfo(True, "Play notification sound after image generation").info("notification.mp3 should be present in the root directory").needs_reload_ui(),
            "notification_volume": OptionInfo(100, "Notification sound volume", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1}).info("in %"),
        },
    )
)

options_templates.update(
    options_section(
        ("saving-paths", "Paths for saving", "saving"),
        {
            "outdir_samples": OptionInfo("", "Output directory for images", component_args=hide_dirs).info("if empty, defaults to the <b>3</b> settings below"),
            "outdir_txt2img_samples": OptionInfo(util.truncate_path(os.path.join(default_output_dir, "txt2img-images")), "Output directory for txt2img images", component_args=hide_dirs),
            "outdir_img2img_samples": OptionInfo(util.truncate_path(os.path.join(default_output_dir, "img2img-images")), "Output directory for img2img images", component_args=hide_dirs),
            "outdir_extras_samples": OptionInfo(util.truncate_path(os.path.join(default_output_dir, "extras-images")), "Output directory for images from extras tab", component_args=hide_dirs),
            "outdir_grids": OptionInfo("", "Output directory for grids", component_args=hide_dirs).info("if empty, defaults to the <b>2</b> settings below"),
            "outdir_txt2img_grids": OptionInfo(util.truncate_path(os.path.join(default_output_dir, "txt2img-grids")), "Output directory for txt2img grids", component_args=hide_dirs),
            "outdir_img2img_grids": OptionInfo(util.truncate_path(os.path.join(default_output_dir, "img2img-grids")), "Output directory for img2img grids", component_args=hide_dirs),
            "outdir_save": OptionInfo(util.truncate_path(os.path.join(data_path, "log", "images")), "Directory for saving images using the Save button", component_args=hide_dirs),
            "outdir_init_images": OptionInfo(util.truncate_path(os.path.join(default_output_dir, "init-images")), "Directory for saving init images when using img2img", component_args=hide_dirs),
        },
    )
)

options_templates.update(
    options_section(
        ("saving-to-dirs", "Saving to a directory", "saving"),
        {
            "save_to_dirs": OptionInfo(True, "Save images to a subdirectory"),
            "grid_save_to_dirs": OptionInfo(True, "Save grids to a subdirectory"),
            "use_save_to_dirs_for_ui": OptionInfo(False, 'When using "Save" button, save images to a subdirectory'),
            "directories_filename_pattern": OptionInfo("[date]", "Directory name pattern", component_args=hide_dirs).link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Images-Filename-Name-and-Subdirectory"),
            "directories_max_prompt_words": OptionInfo(8, "Max prompt words for [prompt_words] pattern", gr.Slider, {"minimum": 1, "maximum": 20, "step": 1, **hide_dirs}),
        },
    )
)

options_templates.update(
    options_section(
        ("upscaling", "Upscaling", "postprocessing"),
        {
            "ESRGAN_tile": OptionInfo(256, "Tile Size for Upscalers", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}).info("0 = no tiling"),
            "ESRGAN_tile_overlap": OptionInfo(16, "Tile Overlap for Upscalers", gr.Slider, {"minimum": 0, "maximum": 64, "step": 4}).info("Low values = visible seam"),
            "upscaler_for_img2img": OptionInfo(None, "Upscaler for img2img", gr.Dropdown, lambda: {"choices": [x.name for x in shared.sd_upscalers]}),
        },
    )
)

options_templates.update(
    options_section(
        ("face-restoration", "Face restoration", "postprocessing"),
        {
            "face_restoration": OptionInfo(False, "Restore faces", infotext="Face restoration").info("will use a third-party model on generation result to reconstruct faces"),
            "face_restoration_model": OptionInfo("CodeFormer", "Face restoration model", gr.Radio, lambda: {"choices": [x.name() for x in shared.face_restorers]}),
            "code_former_weight": OptionInfo(0.5, "CodeFormer weight", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}).info("0 = maximum effect; 1 = minimum effect"),
            "face_restoration_unload": OptionInfo(False, "Move face restoration model from VRAM into RAM after processing"),
        },
    )
)

options_templates.update(
    options_section(
        ("system", "System", "system"),
        {
            "auto_launch_browser": OptionInfo("Local", "Automatically open webui in browser on startup", gr.Radio, lambda: {"choices": ["Disable", "Local", "Remote"]}),
            "enable_console_prompts": OptionInfo(False, "Print prompts to console when generating with txt2img and img2img."),
            "show_warnings": OptionInfo(False, "Show warnings in console.").needs_reload_ui(),
            "show_gradio_deprecation_warnings": OptionInfo(True, "Show gradio deprecation warnings in console.").needs_reload_ui(),
            "memmon_poll_rate": OptionInfo(8, "VRAM usage polls per second during generation.", gr.Slider, {"minimum": 0, "maximum": 40, "step": 1}).info("0 = disable"),
            "samples_log_stdout": OptionInfo(False, "Always print all generation info to standard output"),
            "multiple_tqdm": OptionInfo(True, "Add a second progress bar to the console that shows progress for an entire job."),
            "enable_upscale_progressbar": OptionInfo(True, "Show a progress bar in the console for tiled upscaling."),
            "list_hidden_files": OptionInfo(True, "Load models/files in hidden directories").info('directory is hidden if its name starts with "."'),
            "disable_mmap_load_safetensors": OptionInfo(False, "Disable memmapping for loading .safetensors files.").info("fixes very slow loading speed in some cases"),
            "hide_ldm_prints": OptionInfo(True, "Prevent Stability-AI's ldm/sgm modules from printing noise to console."),
            "dump_stacks_on_signal": OptionInfo(False, "Print stack traces before exiting the program with ctrl+c."),
        },
    )
)

options_templates.update(
    options_section(
        ("API", "API", "system"),
        {
            "api_enable_requests": OptionInfo(True, "Allow http:// and https:// URLs for input images in API", restrict_api=True),
            "api_forbid_local_requests": OptionInfo(True, "Forbid URLs to local resources", restrict_api=True),
            "api_useragent": OptionInfo("", "User agent for requests", restrict_api=True),
        },
    )
)

options_templates.update(
    options_section(
        ("sd", "Stable Diffusion", "sd"),
        {
            "sd_model_checkpoint": OptionInfo(None, "Stable Diffusion checkpoint", gr.Dropdown, lambda: {"choices": shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short)}, refresh=shared_items.refresh_checkpoints, infotext="Model hash"),
            "sd_checkpoint_cache": OptionInfo(0, "Number of checkpoints to cache in RAM", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
            "sd_unet": OptionInfo("Automatic", "SD Unet", gr.Dropdown, lambda: {"choices": shared_items.sd_unet_items()}, refresh=shared_items.refresh_unet_list).info("choose Unet model: Automatic = use one with same filename as checkpoint; None = use Unet from checkpoint"),
            "emphasis": OptionInfo("Original", "Emphasis mode", gr.Radio, lambda: {"choices": [x.name for x in sd_emphasis.options]}, infotext="Emphasis").info("makes it possible to make model to pay (more:1.1) or (less:0.9) attention to text when you use the syntax in prompt; " + sd_emphasis.get_options_descriptions()),
            "enable_batch_seeds": OptionInfo(True, "Make K-diffusion samplers produce same images in a batch as when making a single image"),
            "comma_padding_backtrack": OptionInfo(20, "Prompt word wrap length limit", gr.Slider, {"minimum": 0, "maximum": 74, "step": 1}).info("in tokens - for texts shorter than specified, if they don't fit into 75 token limit, move them to the next 75 token chunk"),
            "CLIP_stop_at_last_layers": OptionInfo(1, "Clip skip", gr.Slider, {"minimum": 1, "maximum": 12, "step": 1}, infotext="Clip skip").link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#clip-skip").info("ignore last layers of CLIP network; 1 = disable, 2 = skip one layer, etc."),
            "clip_skip_sdxl": OptionInfo(False, "Override Clip Skip for SDXL"),
            "upcast_attn": OptionInfo(False, "Upcast cross attention layer to float32"),
            "randn_source": OptionInfo("GPU", "Random number generator source.", gr.Radio, {"choices": ["GPU", "CPU", "NV"]}, infotext="RNG").info("changes seeds drastically; use CPU to produce the same picture across different videocard vendors; use NV to produce same picture as on NVidia videocards"),
            "tiling": OptionInfo(False, "Tiling", infotext="Tiling").info("produce a tileable picture"),
            "hires_fix_refiner_pass": OptionInfo("second pass", "Hires fix: which pass to enable refiner for", gr.Radio, {"choices": ["first pass", "second pass", "both passes"]}, infotext="Hires refiner"),
        },
    )
)

options_templates.update(
    options_section(
        ("sdxl", "Stable Diffusion XL", "sd"),
        {
            "sdxl_crop_top": OptionInfo(0, "crop top coordinate"),
            "sdxl_crop_left": OptionInfo(0, "crop left coordinate"),
            "sdxl_refiner_low_aesthetic_score": OptionInfo(2.5, "SDXL low aesthetic score", gr.Number).info("used for refiner model negative prompt"),
            "sdxl_refiner_high_aesthetic_score": OptionInfo(6.0, "SDXL high aesthetic score", gr.Number).info("used for refiner model prompt"),
        },
    )
)

options_templates.update(
    options_section(
        ("vae", "VAE", "sd"),
        {
            "sd_vae_explanation": OptionHTML(
                """
<abbr title='Variational autoencoder'>VAE</abbr> is a neural network that transforms a standard <abbr title='red/green/blue'>RGB</abbr>
image into latent space representation and back. Latent space representation is what stable diffusion is working on during sampling
(i.e. when the progress bar is between empty and full). For txt2img, VAE is used to create a resulting image after the sampling is finished.
For img2img, VAE is used to process user's input image before the sampling, and to create an image after sampling.
"""
            ),
            "sd_vae_checkpoint_cache": OptionInfo(0, "VAE Checkpoints to cache in RAM", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}).needs_restart(),
            "sd_vae": OptionInfo("Automatic", "SD VAE", gr.Dropdown, lambda: {"choices": shared_items.sd_vae_items()}, refresh=shared_items.refresh_vae_list, infotext="VAE").info("choose VAE model: Automatic = use one with same filename as checkpoint; None = use VAE from checkpoint"),
            "sd_vae_overrides_per_model_preferences": OptionInfo(True, "Selected VAE overrides per-model preferences").info("you can set per-model VAE either by editing user metadata for checkpoints, or by making the VAE have same name as checkpoint"),
            "prefer_vae_precision_float16": OptionInfo(False, "Prefer VAE in float16 precision").info("most VAE at half precision produces NaNs; <b>enable with caution</b>"),
            "auto_vae_precision_bfloat16": OptionInfo(False, "Automatically convert VAE to bfloat16").info("triggers when a tensor with NaNs is produced in VAE; disabling the option in this case will result in a black square image; if enabled, overrides the option below"),
            "auto_vae_precision": OptionInfo(True, "Automatically revert VAE to 32-bit floats").info("triggers when a tensor with NaNs is produced in VAE; disabling the option in this case will result in a black square image"),
            "sd_vae_encode_method": OptionInfo("Full", "VAE type for encode", gr.Radio, {"choices": ["Full", "TAESD"]}, infotext="VAE Encoder").info("method to encode image to latent (use in img2img, hires-fix or inpaint mask)"),
            "sd_vae_decode_method": OptionInfo("Full", "VAE type for decode", gr.Radio, {"choices": ["Full", "TAESD"]}, infotext="VAE Decoder").info("method to decode latent to image"),
        },
    )
)

options_templates.update(
    options_section(
        ("img2img", "img2img", "sd"),
        {
            "inpainting_mask_weight": OptionInfo(1.0, "Inpainting conditioning mask strength", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext="Conditional mask weight"),
            "initial_noise_multiplier": OptionInfo(1.0, "Noise multiplier for img2img", gr.Slider, {"minimum": 0.0, "maximum": 1.5, "step": 0.001}, infotext="Noise multiplier"),
            "img2img_extra_noise": OptionInfo(0.0, "Extra noise multiplier for img2img and hires fix", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext="Extra noise").info("0 = disabled (default); should be lower than denoising strength"),
            "img2img_color_correction": OptionInfo(False, "Apply color correction to img2img results to match original colors."),
            "img2img_fix_steps": OptionInfo(False, "With img2img, do exactly the amount of steps the slider specifies.").info("normally you'd do less with less denoising"),
            "img2img_background_color": OptionInfo("#ffffff", "With img2img, fill transparent parts of the input image with this color.", ui_components.FormColorPicker, {}),
            "img2img_editor_height": OptionInfo(720, "Height of the image editor", gr.Slider, {"minimum": 80, "maximum": 1600, "step": 1}).info("in pixels").needs_reload_ui(),
            "img2img_sketch_default_brush_color": OptionInfo("#ffffff", "Sketch initial brush color", ui_components.FormColorPicker, {}).info("default brush color of img2img sketch").needs_reload_ui(),
            "img2img_inpaint_mask_brush_color": OptionInfo("#ffffff", "Inpaint mask brush color", ui_components.FormColorPicker, {}).info("brush color of inpaint mask").needs_reload_ui(),
            "img2img_inpaint_sketch_default_brush_color": OptionInfo("#ffffff", "Inpaint sketch initial brush color", ui_components.FormColorPicker, {}).info("default brush color of img2img inpaint sketch").needs_reload_ui(),
            "return_mask": OptionInfo(False, "For inpainting, include the greyscale mask in results for web"),
            "return_mask_composite": OptionInfo(False, "For inpainting, include masked composite in results for web"),
            "img2img_batch_show_results_limit": OptionInfo(32, "Show the first N batch img2img results in UI", gr.Slider, {"minimum": -1, "maximum": 1000, "step": 1}).info("0: disable, -1: show all images. Too many images can cause lag"),
            "overlay_inpaint": OptionInfo(True, "Overlay original for inpaint").info("when inpainting, overlay the original image over the areas that weren't inpainted."),
        },
    )
)

options_templates.update(
    options_section(
        ("optimizations", "Optimizations", "sd"),
        {
            "cross_attention_optimization": OptionInfo("Automatic", "Cross attention optimization", gr.Dropdown, lambda: {"choices": ["Automatic"], "interactive": False}),
            "s_min_uncond": OptionInfo(0.0, "Negative Guidance minimum sigma", gr.Slider, {"minimum": 0.0, "maximum": 8.0, "step": 0.05}).link("PR", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9177").info("skip negative prompt for some steps when the image is almost ready; 0=disable; higher=faster"),
            "skip_early_cond": OptionInfo(0.0, "Ignore negative prompt during early steps", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.05}, infotext="Skip Early CFG").info("disables CFG on a proportion of steps at the beginning of generation; 0=disable; 1=skip all; can both improve sample diversity/quality and speed up sampling"),
            "persistent_cond_cache": OptionInfo(True, "Persistent cond cache").info("do not recalculate conds from prompts if prompts have not changed since previous calculation"),
            "fp8_storage": OptionInfo(False, "Store UNet Weights in fp8").info("store the weights in fp8; inference in fp16; reduce memory usage; reduce speed; reduce quality").needs_restart(),
            "fp8_fast": OptionInfo(False, "Inference UNet in fast fp8 operations").info("inference in fp8 using <b>torch._scaled_mm</b>; increase speed; reduce quality; require <b>RTX 40</b> or later").needs_restart(),
            "cublas_fast": OptionInfo(False, "Inference UNet in fast cublas operations").info("inference using <b>CublasLinear</b>; increase speed; require <b>manual</b> installation first").needs_restart(),
            "token_merging_explanation": OptionHTML(
                """
<b>Token Merging</b> speeds up the diffusion process by fusing "redundant" tokens together, but also reduces quality as a result.
[<a href="https://github.com/dbolya/tomesd">GitHub</a>] <br>
<b>Note:</b> Has no effect on SDXL when Max Downsample is set to 1
                """
            ),
            "token_merging_ratio": OptionInfo(0.0, "Token merging ratio", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.05}, infotext="Token merging ratio").info("0=disable; higher=faster"),
            "token_merging_ratio_img2img": OptionInfo(0.0, "Token merging ratio for img2img", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.05}).info("overrides base ratio if non-zero"),
            "token_merging_ratio_hr": OptionInfo(0.0, "Token merging ratio for Hires. fix", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.05}, infotext="Token merging ratio hr").info("overrides base ratio if non-zero"),
            "token_merging_stride": OptionInfo(2, "Token merging - Stride", gr.Slider, {"minimum": 1, "maximum": 8, "step": 1}).info("higher=faster"),
            "token_merging_downsample": OptionInfo(1, "Token merging - Max Downsample", gr.Slider, {"minimum": 1, "maximum": 4, "step": 1}).info("higher=faster"),
            "token_merging_no_rand": OptionInfo(False, "Token merging - No random").info("reduce randomness by always selecting the same corner"),
        },
    )
)

options_templates.update(
    options_section(
        ("compatibility", "Compatibility", "sd"),
        {
            "use_old_karras_scheduler_sigmas": OptionInfo(False, "Use old karras scheduler sigmas (0.1 to 10)."),
            "no_dpmpp_sde_batch_determinism": OptionInfo(False, "Do not make DPM++ SDE deterministic across different batch sizes."),
            "use_old_hires_fix_width_height": OptionInfo(False, "For hires fix, use width/height sliders to set final resolution rather than first pass (disables Upscale by, Resize width/height to)."),
            "hires_fix_use_firstpass_conds": OptionInfo(False, "For hires fix, calculate conds of second pass using extra networks of first pass."),
            "use_old_scheduling": OptionInfo(False, "Use old prompt editing timelines.", infotext="Old prompt editing timelines").info("For [red:green:N]; old: If N < 1, it's a fraction of steps (and hires fix uses range from 0 to 1), if N >= 1, it's an absolute number of steps; new: If N has a decimal point in it, it's a fraction of steps (and hires fix uses range from 1 to 2), othewrwise it's an absolute number of steps"),
            "use_downcasted_alpha_bar": OptionInfo(False, "Downcast model alphas_cumprod to fp16 before sampling. For reproducing old seeds.", infotext="Downcast alphas_cumprod"),
        },
    )
)

options_templates.update(
    options_section(
        ("extra_networks", "Extra Networks", "sd"),
        {
            "extra_networks_tree_view_explanation": OptionHTML(
                """
<b>Tree View</b> is a rather... "controversial" feature,
which adds a folder structure to visualize the ExtraNetworks.
However, the resulting UI is quite... sluggish.
                """
            ),
            "extra_networks_tree_view_enable": OptionInfo(False, "Enable the Tree View").needs_reload_ui(),
            "extra_networks_tree_view_default_enabled": OptionInfo(False, "Show the Tree View by default").needs_reload_ui(),
            "extra_networks_show_hidden_directories": OptionInfo(False, "Show hidden directories").info("directory is hidden if its name starts with a ."),
            "extra_networks_dir_button_function": OptionInfo(False, "Add a '/' to the beginning of directory buttons").info("Buttons will display the contents of the selected directory without acting as a search filter"),
            "extra_networks_hidden_models": OptionInfo("When searched", "Show cards for models in hidden directories", gr.Radio, {"choices": ["Always", "When searched", "Never"]}).info('"When searched" option will only show the item when the search string contains 4 characters or more'),
            "extra_networks_default_multiplier": OptionInfo(1.0, "Default multiplier for extra networks", gr.Slider, {"minimum": 0.0, "maximum": 2.0, "step": 0.01}),
            "extra_networks_card_width": OptionInfo(0, "Card width for Extra Networks").info("in pixels"),
            "extra_networks_card_height": OptionInfo(0, "Card height for Extra Networks").info("in pixels"),
            "extra_networks_card_text_scale": OptionInfo(1.0, "Card text scale", gr.Slider, {"minimum": 0.0, "maximum": 2.0, "step": 0.01}).info("1 = original size"),
            "extra_networks_card_show_desc": OptionInfo(True, "Show description on card"),
            "extra_networks_card_description_is_html": OptionInfo(False, "Treat card description as HTML"),
            "extra_networks_card_order_field": OptionInfo("Path", "Default order field for Extra Networks cards", gr.Dropdown, {"choices": ["Path", "Name", "Date Created", "Date Modified"]}).needs_reload_ui(),
            "extra_networks_card_order": OptionInfo("Ascending", "Default order for Extra Networks cards", gr.Dropdown, {"choices": ["Ascending", "Descending"]}).needs_reload_ui(),
            "extra_networks_add_text_separator": OptionInfo(" ", "Extra networks separator").info("extra text to add before <...> when adding extra network to prompt"),
            "ui_extra_networks_tab_reorder": OptionInfo("", "Extra networks tab order").needs_reload_ui(),
            "textual_inversion_add_hashes_to_infotext": OptionInfo(True, "Add Textual Inversion hashes to infotext"),
        },
    )
)

options_templates.update(
    options_section(
        ("ui_prompt_editing", "Prompt editing", "ui"),
        {
            "keyedit_precision_attention": OptionInfo(0.1, "Precision for (attention:1.1) when editing the prompt with Ctrl+up/down", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001}),
            "keyedit_precision_extra": OptionInfo(0.05, "Precision for <extra networks:0.9> when editing the prompt with Ctrl+up/down", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001}),
            "keyedit_delimiters": OptionInfo(r".,\/!?%^*;:{}=`~() ", "Word delimiters when editing the prompt with Ctrl+up/down"),
            "keyedit_delimiters_whitespace": OptionInfo(["Tab", "Carriage Return", "Line Feed"], "Ctrl+up/down whitespace delimiters", gr.CheckboxGroup, lambda: {"choices": ["Tab", "Carriage Return", "Line Feed"]}),
            "keyedit_move": OptionInfo(True, "Alt+left/right moves prompt elements"),
            "disable_token_counters": OptionInfo(False, "Disable prompt token counters"),
            "include_styles_into_token_counters": OptionInfo(True, "Count tokens of enabled styles").info("When calculating how many tokens the prompt has, also consider tokens added by enabled styles."),
        },
    )
)

options_templates.update(
    options_section(
        ("ui_gallery", "Gallery", "ui"),
        {
            "return_grid": OptionInfo(True, "Show grid in gallery"),
            "do_not_show_images": OptionInfo(False, "Do not show any images in gallery"),
            "js_modal_lightbox": OptionInfo(True, "Full page image viewer: enable"),
            "js_modal_lightbox_initially_zoomed": OptionInfo(True, "Full page image viewer: show images zoomed in by default"),
            "js_modal_lightbox_gamepad": OptionInfo(False, "Full page image viewer: navigate with gamepad"),
            "js_modal_lightbox_gamepad_repeat": OptionInfo(250, "Full page image viewer: gamepad repeat period").info("in milliseconds"),
            "sd_webui_modal_lightbox_icon_opacity": OptionInfo(1, "Full page image viewer: control icon unfocused opacity", gr.Slider, {"minimum": 0.0, "maximum": 1, "step": 0.01}, onchange=shared.reload_gradio_theme).info("for mouse only").needs_reload_ui(),
            "sd_webui_modal_lightbox_toolbar_opacity": OptionInfo(0.9, "Full page image viewer: tool bar opacity", gr.Slider, {"minimum": 0.0, "maximum": 1, "step": 0.01}, onchange=shared.reload_gradio_theme).info("for mouse only").needs_reload_ui(),
            "gallery_height": OptionInfo("", "Gallery height", gr.Textbox).info("can be any valid CSS value, for example 768px or 20em").needs_reload_ui(),
            "open_dir_button_choice": OptionInfo("Subdirectory", "What directory the [📂] button opens", gr.Radio, {"choices": ["Output Root", "Subdirectory", "Subdirectory (even temp dir)"]}),
        },
    )
)

options_templates.update(
    options_section(
        ("ui_alternatives", "UI alternatives", "ui"),
        {
            "show_scheduler": OptionInfo(False, "Display the Scheduler Options").needs_reload_ui(),
            "show_refiner": OptionInfo(False, "Display the Refiner Accordions").info("(deprecated) feature for SDXL").needs_reload_ui(),
            "show_rescale_cfg": OptionInfo(False, "Display the Rescale CFG Slider").info("feature for v-pred checkpoints").needs_reload_ui(),
            "show_mahiro": OptionInfo(False, "Display the MaHiRo toggle").info('see <a href="https://huggingface.co/spaces/yoinked/blue-arxiv">blue-arxiv</a> - <b>id:</b> <ins>2024-1208.1</ins>').needs_reload_ui(),
            "compact_prompt_box": OptionInfo(False, "Compact prompt layout").info("puts prompt and negative prompt inside the Generate tab, leaving more vertical space for the image on the right").needs_reload_ui(),
            "dimensions_and_batch_together": OptionInfo(True, "Show Width/Height and Batch sliders in same row").needs_reload_ui(),
            "sd_checkpoint_dropdown_use_short": OptionInfo(False, "Checkpoint dropdown: use filenames without paths").info("models in subdirectories like photo/sd15.ckpt will be listed as just sd15.ckpt"),
            "hires_fix_show_sampler": OptionInfo(False, "Hires fix: show hires checkpoint and sampler selection").needs_reload_ui(),
            "hires_fix_show_prompts": OptionInfo(False, "Hires fix: show hires prompt and negative prompt").needs_reload_ui(),
            "txt2img_settings_accordion": OptionInfo(False, "Settings in txt2img hidden under Accordion").needs_reload_ui(),
            "img2img_settings_accordion": OptionInfo(False, "Settings in img2img hidden under Accordion").needs_reload_ui(),
            "interrupt_after_current": OptionInfo(True, "Don't Interrupt in the middle").info("when using Interrupt button, if generating more than one image, stop after the generation of an image has finished, instead of immediately"),
        },
    )
)

options_templates.update(
    options_section(
        ("ui", "User interface", "ui"),
        {
            "localization": OptionInfo("None", "Localization", gr.Dropdown, lambda: {"choices": ["None"] + list(localization.localizations.keys())}, refresh=lambda: localization.list_localizations(cmd_opts.localizations_dir)).needs_reload_ui(),
            "quicksettings_list": OptionInfo(["sd_model_checkpoint", "sd_vae", "CLIP_stop_at_last_layers"], "Quicksettings list", ui_components.DropdownMulti, lambda: {"choices": list(shared.opts.data_labels.keys())}).js("info", "settingsHintsShowQuicksettings").info("setting entries that appear at the top of page rather than in settings tab").needs_reload_ui(),
            "ui_tab_order": OptionInfo([], "UI tab order", ui_components.DropdownMulti, lambda: {"choices": list(shared.tab_names)}).needs_reload_ui(),
            "hidden_tabs": OptionInfo([], "Hidden UI tabs", ui_components.DropdownMulti, lambda: {"choices": list(shared.tab_names)}).needs_reload_ui(),
            "ui_reorder_list": OptionInfo([], "UI item order for txt2img/img2img tabs", ui_components.DropdownMulti, lambda: {"choices": list(shared_items.ui_reorder_categories())}).info("selected items appear first").needs_reload_ui(),
            "gradio_theme": OptionInfo("Default", "Gradio theme", ui_components.DropdownEditable, lambda: {"choices": ["Default"] + shared_gradio_themes.gradio_hf_hub_themes}).info("you can also manually enter any of themes from the <a href='https://huggingface.co/spaces/gradio/theme-gallery'>gallery</a>.").needs_reload_ui(),
            "gradio_themes_cache": OptionInfo(True, "Cache gradio themes locally").info("disable to update the selected Gradio theme"),
            "show_progress_in_title": OptionInfo(True, "Show generation progress in window title."),
            "send_seed": OptionInfo(True, "Send seed when sending prompt or image to other interface"),
            "send_size": OptionInfo(True, "Send size when sending prompt or image to another interface"),
            "paste_safe_guard": OptionInfo(True, 'Disable the "Read generation parameters" button when negative prompt contains texts'),
        },
    )
)


options_templates.update(
    options_section(
        ("infotext", "Infotext", "ui"),
        {
            "infotext_explanation": OptionHTML(
                """
Infotext is what this software calls the text that contains generation parameters and can be used to generate the same picture again.
It is displayed in UI below the image. To use infotext, paste it into the prompt and click the ↙️ paste button.
"""
            ),
            "enable_pnginfo": OptionInfo(True, "Write infotext to metadata of the generated image"),
            "save_txt": OptionInfo(False, "Create a text file with infotext next to every generated image"),
            "add_model_name_to_info": OptionInfo(True, "Add model name to infotext"),
            "add_model_hash_to_info": OptionInfo(True, "Add model hash to infotext"),
            "add_vae_name_to_info": OptionInfo(True, "Add VAE name to infotext"),
            "add_vae_hash_to_info": OptionInfo(True, "Add VAE hash to infotext"),
            "add_user_name_to_info": OptionInfo(False, "Add user name to infotext when authenticated"),
            "add_version_to_infotext": OptionInfo(True, "Add program version to infotext"),
            "disable_weights_auto_swap": OptionInfo(True, "Disregard checkpoint information from pasted infotext").info("when reading generation parameters from text into UI"),
            "infotext_skip_pasting": OptionInfo([], "Disregard fields from pasted infotext", ui_components.DropdownMulti, lambda: {"choices": shared_items.get_infotext_names()}),
            "infotext_styles": OptionInfo("Apply if any", "Infer styles from prompts of pasted infotext", gr.Radio, {"choices": ["Ignore", "Apply", "Discard", "Apply if any"]})
            .info("when reading generation parameters from text into UI)")
            .html(
                """<ul style='margin-left: 1.5em'>
<li>Ignore: keep prompt and styles dropdown as it is.</li>
<li>Apply: remove style text from prompt, always replace styles dropdown value with found styles (even if none are found).</li>
<li>Discard: remove style text from prompt, keep styles dropdown as it is.</li>
<li>Apply if any: remove style text from prompt; if any styles are found in prompt, put them into styles dropdown, otherwise keep it as it is.</li>
</ul>"""
            ),
        },
    )
)

options_templates.update(
    options_section(
        ("ui", "Live previews", "ui"),
        {
            "show_progressbar": OptionInfo(True, "Show progressbar"),
            "live_previews_enable": OptionInfo(True, "Show live previews of the created image"),
            "live_previews_image_format": OptionInfo("jpeg", "Live preview file format", gr.Radio, {"choices": ["jpeg", "png", "webp"]}),
            "show_progress_grid": OptionInfo(True, "Show previews of all images generated in a batch as a grid"),
            "show_progress_every_n_steps": OptionInfo(10, "Live preview display period", gr.Slider, {"minimum": -1, "maximum": 32, "step": 1}).info("in sampling steps - show new live preview image every N sampling steps; -1 = only show after completion of batch"),
            "show_progress_type": OptionInfo("TAESD", "Live preview method", gr.Radio, {"choices": ["Approx NN", "Approx cheap", "TAESD", "RGB"]}).info("<b>Approx NN</b>: legacy preview method; <b>TAESD</b>: high quality preview method; <b>Approx cheap</b> / <b>RGB</b>: fast but low quality preview method").info("<b>Approx NN</b> and <b>TAESD</b> will download additional model"),
            "live_preview_content": OptionInfo("Prompt", "Live preview subject", gr.Radio, {"choices": ["Combined", "Prompt", "Negative prompt"]}),
            "live_preview_refresh_period": OptionInfo(1000, "Progressbar and preview update period").info("in milliseconds"),
            "live_preview_fast_interrupt": OptionInfo(False, "Return image with chosen live preview method on interrupt").info("makes interrupts faster"),
            "js_live_preview_in_modal_lightbox": OptionInfo(False, "Show Live preview in full page image viewer"),
        },
    )
)

options_templates.update(
    options_section(
        ("sampler-params", "Sampler parameters", "sd"),
        {
            "hide_samplers": OptionInfo([], "Hide samplers in user interface", ui_components.DropdownMulti, lambda: {"choices": [x.name for x in shared_items.list_samplers()]}).needs_reload_ui(),
            "hide_samplers_invert": OptionInfo(False, "Invert the above setting").info("<b>ie.</b> it is now a whitelist instead of a blacklist").needs_reload_ui(),
            "eta_ddim": OptionInfo(0.0, "Eta for DDIM", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext="Eta DDIM").info("noise multiplier; higher = more unpredictable results"),
            "eta_ancestral": OptionInfo(1.0, "Eta for k-diffusion samplers", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext="Eta").info("noise multiplier; currently only applies to ancestral samplers (i.e. Euler a) and SDE samplers"),
            "ddim_discretize": OptionInfo("uniform", "img2img DDIM discretize", gr.Radio, {"choices": ["uniform", "quad"]}),
            "s_churn": OptionInfo(0.0, "sigma churn", gr.Slider, {"minimum": 0.0, "maximum": 100.0, "step": 0.01}, infotext="Sigma churn").info("amount of stochasticity; only applies to Euler, Heun, and DPM2"),
            "s_tmin": OptionInfo(0.0, "sigma tmin", gr.Slider, {"minimum": 0.0, "maximum": 10.0, "step": 0.01}, infotext="Sigma tmin").info("enable stochasticity; start value of the sigma range; only applies to Euler, Heun, and DPM2"),
            "s_tmax": OptionInfo(0.0, "sigma tmax", gr.Slider, {"minimum": 0.0, "maximum": 999.0, "step": 0.01}, infotext="Sigma tmax").info("0 = inf; end value of the sigma range; only applies to Euler, Heun, and DPM2"),
            "s_noise": OptionInfo(1.0, "sigma noise", gr.Slider, {"minimum": 0.0, "maximum": 1.1, "step": 0.001}, infotext="Sigma noise").info("amount of additional noise to counteract loss of detail during sampling"),
            "sigma_min": OptionInfo(0.0, "sigma min", gr.Number, infotext="Schedule min sigma").info("0 = default (~0.03); minimum noise strength for k-diffusion noise scheduler"),
            "sigma_max": OptionInfo(0.0, "sigma max", gr.Number, infotext="Schedule max sigma").info("0 = default (~14.6); maximum noise strength for k-diffusion noise scheduler"),
            "rho": OptionInfo(0.0, "rho", gr.Number, infotext="Schedule rho").info("0 = default (7 for karras, 1 for polyexponential); higher values result in a steeper noise schedule (decreases faster)"),
            "eta_noise_seed_delta": OptionInfo(0, "Eta noise seed delta", gr.Number, {"precision": 0}, infotext="ENSD").info("ENSD; does not improve anything, just produces different results for ancestral samplers - only useful for reproducing images"),
            "always_discard_next_to_last_sigma": OptionInfo(False, "Always discard next-to-last sigma", infotext="Discard penultimate sigma").link("PR", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/6044"),
            "sgm_noise_multiplier": OptionInfo(False, "SGM noise multiplier", infotext="SGM noise multiplier").link("PR", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12818").info("Match initial noise to official SDXL implementation - only useful for reproducing images"),
            "uni_pc_variant": OptionInfo("bh1", "UniPC variant", gr.Radio, {"choices": ["bh1", "bh2", "vary_coeff"]}, infotext="UniPC variant"),
            "uni_pc_skip_type": OptionInfo("time_uniform", "UniPC skip type", gr.Radio, {"choices": ["time_uniform", "time_quadratic", "logSNR"]}, infotext="UniPC skip type"),
            "uni_pc_order": OptionInfo(3, "UniPC order", gr.Slider, {"minimum": 1, "maximum": 50, "step": 1}, infotext="UniPC order").info("must be < sampling steps"),
            "uni_pc_lower_order_final": OptionInfo(True, "UniPC lower order final", infotext="UniPC lower order final"),
            "sd_noise_schedule": OptionInfo("Default", "Noise schedule for sampling", gr.Radio, {"choices": ["Default", "Zero Terminal SNR"]}, infotext="Noise Schedule").info("for use with zero terminal SNR trained models"),
            "beta_dist_alpha": OptionInfo(0.6, "Beta scheduler - alpha", gr.Slider, {"minimum": 0.01, "maximum": 2.0, "step": 0.01}, infotext="Beta scheduler alpha").info("Default = 0.6; the alpha parameter of the beta distribution used in Beta sampling"),
            "beta_dist_beta": OptionInfo(0.6, "Beta scheduler - beta", gr.Slider, {"minimum": 0.01, "maximum": 2.0, "step": 0.01}, infotext="Beta scheduler beta").info("Default = 0.6; the beta parameter of the beta distribution used in Beta sampling"),
        },
    )
)

options_templates.update(
    options_section(
        ("postprocessing", "Postprocessing", "postprocessing"),
        {
            "postprocessing_enable_in_main_ui": OptionInfo([], "Enable postprocessing operations in txt2img and img2img tabs", ui_components.DropdownMulti, lambda: {"choices": [x.name for x in shared_items.postprocessing_scripts()]}),
            "postprocessing_operation_order": OptionInfo([], "Postprocessing operation order", ui_components.DropdownMulti, lambda: {"choices": [x.name for x in shared_items.postprocessing_scripts()]}),
            "upscaling_max_images_in_cache": OptionInfo(5, "Maximum number of images in upscaling cache", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
            "postprocessing_existing_caption_action": OptionInfo("Ignore", "Action for existing captions", gr.Radio, {"choices": ["Ignore", "Keep", "Prepend", "Append"]}).info("when generating captions using postprocessing; Ignore = use generated; Keep = use original; Prepend/Append = combine both"),
        },
    )
)

options_templates.update(
    options_section(
        (None, "Hidden options"),
        {
            "disabled_extensions": OptionInfo([], "Disable these extensions"),
            "disable_all_extensions": OptionInfo("none", "Disable all extensions (preserves the list of disabled extensions)", gr.Radio, {"choices": ["none", "extra", "all"]}),
            "restore_config_state_file": OptionInfo("", "Config state file to restore from, under 'config-states/' folder"),
            "sd_checkpoint_hash": OptionInfo("", "SHA256 hash of the current checkpoint"),
        },
    )
)
