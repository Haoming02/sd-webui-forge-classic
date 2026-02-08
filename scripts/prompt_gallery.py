import copy
import os
import random
import sys
import traceback
import shlex
import yaml
import platform
import subprocess as sp
import shutil
import json
import tempfile
import gradio as gr
import csv
import typing
import base64
import io
from PIL import Image
import mimetypes
from modules.paths_internal import extensions_dir
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')

import modules.generation_parameters_copypaste as parameters_copypaste
from modules.generation_parameters_copypaste import image_from_url_text
import modules.scripts as scripts
from modules.processing import Processed, process_images

from modules.shared import opts, cmd_opts, OptionInfo, hide_dirs, state
import modules.shared as shared

if '__file__' in locals().keys():
    root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
else:
    root_path = os.path.abspath(shared.script_path)

rela_path =  os.path.join('extensions')

try:
    with open(os.path.join(extensions_dir, 'prompt_gallery_name.json')) as fd:
        extension_name = json.load(fd)['name']
except:
    extension_name = "Prompt Gallery"
    
OUTPATH_SAMPLES = os.path.join(extensions_dir, extension_name, 'assets', 'preview')
OUTPATH_GRIDS =  os.path.join(extensions_dir, extension_name, 'assets', 'grid')

BATCH_SIZE = 2
N_ITER = 2
STEPS = 30
CFG_SCALE = 11.5
WIDTH = 512
HEIGHT = 768
SAMPLER_INDEX = 1
RESTORE_FACE = 'true'
TILING = 'false'
DO_NOT_SAVE_GRID = 'false'

EXCLUDED_TAGS = ['']
global SKIP_EXISTS
SKIP_EXISTS = True

OUTPUTS_DICT = list()
OUTPUTS = {}
rawDict = {}
qc_dict = {}
trg_img = ''
current_folder = ''


def options_section_def(section_identifier, options_dict):
    ret = {}
    for k, v in options_dict.items():
        v.section = section_identifier
        ret[k] = v.default
    return ret

pg_templates = {}

pg_templates.update(options_section_def(('saving-images', "Saving images/grids"), {
    "samples_save": OptionInfo(True, "Always save all generated images"),
    "samples_format": OptionInfo('png', 'File format for images'),
    "samples_filename_pattern": OptionInfo("[seed]", "Images filename pattern", component_args=hide_dirs),
    "save_images_add_number": OptionInfo(True, "Add number to filename when saving", component_args=hide_dirs),

    "grid_save": OptionInfo(True, "Always save all generated image grids"),
    "grid_format": OptionInfo('png', 'File format for grids'),
    "grid_extended_filename": OptionInfo(False, "Add extended info (seed, prompt) to filename when saving grid"),
    "grid_only_if_multiple": OptionInfo(True, "Do not save grids consisting of one picture"),
    "grid_prevent_empty_spots": OptionInfo(False, "Prevent empty spots in grid (when set to autodetect)"),
    "n_rows": OptionInfo(-1, "Grid row count; use -1 for autodetect and 0 for it to be same as batch size", gr.Slider, {"minimum": -1, "maximum": 16, "step": 1}),

    "enable_pnginfo": OptionInfo(True, "Save text information about generation parameters as chunks to png files"),
    "save_txt": OptionInfo(False, "Create a text file next to every image with generation parameters."),
    "jpeg_quality": OptionInfo(80, "Quality for saved jpeg images", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
    "export_for_4chan": OptionInfo(True, "If PNG image is larger than 4MB or any dimension is larger than 4000, downscale and save copy as JPG"),

    "use_original_name_batch": OptionInfo(False, "Use original name for output filename during batch process in extras tab"),
    "save_selected_only": OptionInfo(True, "When using 'Save' button, only save a single selected image"),
    "do_not_add_watermark": OptionInfo(False, "Do not add watermark to images"),

}))

# pg_templates.update(options_section_def(('saving-paths', "Paths for saving"), {
#     "outdir_samples": OptionInfo("", "Output directory for images; if empty, defaults to three directories below", component_args=hide_dirs),
#     "outdir_txt2img_samples": OptionInfo("outputs/txt2img-images", 'Output directory for txt2img images', component_args=hide_dirs),
#     "outdir_img2img_samples": OptionInfo("outputs/img2img-images", 'Output directory for img2img images', component_args=hide_dirs),
#     "outdir_extras_samples": OptionInfo("outputs/extras-images", 'Output directory for images from extras tab', component_args=hide_dirs),
#     "outdir_grids": OptionInfo("", "Output directory for grids; if empty, defaults to two directories below", component_args=hide_dirs),
#     "outdir_txt2img_grids": OptionInfo("outputs/txt2img-grids", 'Output directory for txt2img grids', component_args=hide_dirs),
#     "outdir_img2img_grids": OptionInfo("outputs/img2img-grids", 'Output directory for img2img grids', component_args=hide_dirs),
#     "outdir_save": OptionInfo("log/images", "Directory for saving images using the Save button", component_args=hide_dirs),
# }))


pg_templates.update(options_section_def(('saving-to-dirs', "Saving to a directory"), {
    "save_to_dirs": OptionInfo(False, "Save images to a subdirectory"),
    "grid_save_to_dirs": OptionInfo(False, "Save grids to a subdirectory"),
    "use_save_to_dirs_for_ui": OptionInfo(False, "When using \"Save\" button, save images to a subdirectory"),
    "directories_filename_pattern": OptionInfo("", "Directory name pattern", component_args=hide_dirs),
    "directories_max_prompt_words": OptionInfo(8, "Max prompt words for [prompt_words] pattern", gr.Slider, {"minimum": 1, "maximum": 20, "step": 1, **hide_dirs}),
}))

pg_templates.update(options_section_def(('upscaling', "Upscaling"), {
    "ESRGAN_tile": OptionInfo(192, "Tile size for ESRGAN upscalers. 0 = no tiling.", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}),
    "ESRGAN_tile_overlap": OptionInfo(8, "Tile overlap, in pixels for ESRGAN upscalers. Low values = visible seam.", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}),
    "realesrgan_enabled_models": OptionInfo(["R-ESRGAN 4x+", "R-ESRGAN 4x+ Anime6B"], "Select which Real-ESRGAN models to show in the web UI. (Requires restart)", gr.CheckboxGroup, lambda: {"choices": realesrgan_models_names()}),
    "upscaler_for_img2img": OptionInfo(None, "Upscaler for img2img", gr.Dropdown, lambda: {"choices": [x.name for x in sd_upscalers]}),
    "use_scale_latent_for_hires_fix": OptionInfo(False, "Upscale latent space image when doing hires. fix"),
}))

pg_templates.update(options_section_def(('ui', "User interface"), {
    "show_progressbar": OptionInfo(True, "Show progressbar"),
    "show_progress_every_n_steps": OptionInfo(0, "Show image creation progress every N sampling steps. Set to 0 to disable. Set to -1 to show after completion of batch.", gr.Slider, {"minimum": -1, "maximum": 32, "step": 1}),
    "show_progress_grid": OptionInfo(True, "Show previews of all images generated in a batch as a grid"),
    "return_grid": OptionInfo(True, "Show grid in results for web"),
    "do_not_show_images": OptionInfo(False, "Do not show any images in results for web"),
    # "disable_weights_auto_swap": OptionInfo(False, "When reading generation parameters from text into UI (from PNG info or pasted text), do not change the selected model/checkpoint."),
    "send_seed": OptionInfo(True, "Send seed when sending prompt or image to other interface"),
    "font": OptionInfo("", "Font for image grids that have text"),
    "js_modal_lightbox": OptionInfo(True, "Enable full page image viewer"),
    "js_modal_lightbox_initially_zoomed": OptionInfo(True, "Show images zoomed in by default in full page image viewer"),
    "show_progress_in_title": OptionInfo(True, "Show generation progress in window title."),
    'localization': OptionInfo("None", "Localization (requires restart)", gr.Dropdown, lambda: {"choices": ["None"] + list(localization.localizations.keys())}, refresh=lambda: localization.list_localizations(cmd_opts.localizations_dir)),
}))

map_sampler_to_idx = {
    'Euler a': 0,
    'Euler': 1,
    'LMS': 2,
    'Heun': 3,
    'DPM2': 4,
    'DPM2 a': 5,
    'DPM fast': 6,
    'DPM adaptive': 7,
    'LMS Karras': 8,
    'DPM2 Karras': 9,
    'DPM2 a Karras': 10,
    'DDIM': 11,
    'PLMS': 12,
    'DPM++ 2S a Karras': 13,
    'DPM++ 2M Karras': 14,
    'DPM++ SDE Karras': 15}

map_keys = {
    "value": "prompt",
    "negative": "negative_prompt"}

map_param = {
    "sd_model": "sd_model",
    "outpath_samples": "outpath_samples",
    "outpath_grids": "outpath_grids",
    "prompt_for_display": "prompt_for_display",
    "styles": "styles",
    "Seed": "seed",
    "Variation seed strength": "subseed_strength",
    "Variation seed": "subseed",
    "seed_resize_from_h": "seed_resize_from_h",
    "seed_resize_from_w": "seed_resize_from_w",
    "Sampler": "sampler_index",
    "batch_size": "batch_size",
    "n_iter": "n_iter",
    "Steps": "steps",
    "CFG scale": "cfg_scale",
    "width": "width",
    "height": "height",
    "restore_faces": "restore_faces",
    "tiling": "tiling",
    "do_not_save_samples": "do_not_save_samples",
    "do_not_save_grid": "do_not_save_grid"}

def process_string_tag(tag):
    return tag


def process_int_tag(tag):
    return int(tag)


def process_float_tag(tag):
    return float(tag)


def process_boolean_tag(tag):
    return True if (tag == "true") else False


prompt_tags = {
    "sd_model": None,
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
    "batch_size": process_int_tag,
    "n_iter": process_int_tag,
    "steps": process_int_tag,
    "cfg_scale": process_float_tag,
    "width": process_int_tag,
    "height": process_int_tag,
    "restore_faces": process_boolean_tag,
    "tiling": process_boolean_tag,
    "do_not_save_samples": process_boolean_tag,
    "do_not_save_grid": process_boolean_tag
}

avatar_prompts = list()
avatar_names = list()
avatar_negatives = list()
avatar_name = ""

def cmdargs(line):
    args = shlex.split(line)
    args[args.index('--outpath_samples')+1] = args[args.index('--outpath_samples')+1][:-1]
    args[args.index('--outpath_grids')+1] = args[args.index('--outpath_grids')+1][:-1]
    pos = 0
    res = {}

    while pos < len(args):
        arg = args[pos]

        assert arg.startswith("--"), f'must start with "--": {arg}'
        tag = arg[2:]

        func = prompt_tags.get(tag, None)
        assert func, f'unknown commandline option: {arg}'

        assert pos+1 < len(args), f'missing argument for command line option {arg}'

        val = args[pos+1]

        res[tag] = func(val)

        pos += 2

    return res



def add_param(key, value, cur_str):
    cur_str += '--{key} {value} '.format(key=key, value=value)
    return cur_str

def parse_size(i_width, i_height, str_size, cur_str):
    i_width = str_size.split('x')[0]
    i_height = str_size.split('x')[1] 
    return i_width, i_height

def parse_virariant_size(str_size, cur_str):
    width = str_size.split('x')[0]
    height = str_size.split('x')[1]
    cur_str = add_param('seed_resize_from_w', width, cur_str)
    cur_str = add_param('seed_resize_from_h', height, cur_str)    
    return cur_str

def parse_param(param_str):
    m_batch_size = BATCH_SIZE
    m_n_iter = N_ITER
    m_steps = STEPS
    m_cfg_scale = CFG_SCALE
    m_width = WIDTH
    m_height = HEIGHT
    m_sampler_index = SAMPLER_INDEX
    # m_tiling = TILING
    m_restore_faces = RESTORE_FACE
    m_do_not_save_grid = DO_NOT_SAVE_GRID
    # m_sd_model = sd_model
    cur_line = ""
    for item in param_str.split(', '):
        if item == '':
            continue
        group = item.split(': ')
        key = group[0]
        value = group[1]
        if key == 'Steps':
            m_steps = value
        elif key == "CFG scale":
            m_cfg_scale = value
        elif key == 'Sampler':
            m_sampler_index = map_sampler_to_idx[value]
        elif key == 'Size':
            m_width, m_height = parse_size(m_width, m_height, value, cur_line)
        elif key == 'Seed resize from':
            cur_line = parse_virariant_size(value, cur_line)
        elif key == 'Seed':
            cur_line = add_param("seed", value, cur_line)
        elif key == 'Variation seed strength':
            cur_line = add_param("subseed_strength", value, cur_line)   
        elif key == 'Variation seed':
            cur_line = add_param("subseed", value, cur_line)   
        # elif key == 'Model hash':
        #     cur_line = add_param("sd_model", m_sd_model, cur_line)   
    cur_line = add_param("batch_size", m_batch_size, cur_line)
    cur_line = add_param("n_iter", m_n_iter, cur_line)
    cur_line = add_param("steps", m_steps, cur_line)
    cur_line = add_param("cfg_scale", m_cfg_scale, cur_line)
    cur_line = add_param("sampler_index", m_sampler_index, cur_line)
    cur_line = add_param("width", m_width, cur_line)
    cur_line = add_param("height", m_height, cur_line)
    cur_line = add_param("restore_faces", m_restore_faces, cur_line)
#     cur_line = add_param("tiling", m_tiling, cur_line)
    cur_line = add_param("do_not_save_grid", m_do_not_save_grid, cur_line)
    return cur_line

def parse_yaml_dict(rawDict, tag, avatar_prompt, avatar_name, default_negative):
    # depth-first-search
    if 'value' in rawDict.keys() or 'negative' in rawDict.keys():
        if SKIP_EXISTS:
            if os.path.exists(os.path.join(OUTPATH_SAMPLES, tag, avatar_name+'.png')) or os.path.exists(os.path.join(OUTPATH_SAMPLES,tag,'Not-available.png')):
                print("Skip "+str(tag))
                return ""
        cur = ""
        parsed_param = False
        m_positive = avatar_prompt
        m_negative = default_negative
        for item in rawDict.items():
            key = item[0]
            value = item[1]
            if key == 'param':
                params = parse_param(rawDict['param'])
                parsed_param = True
            elif key == 'value':
                m_positive = m_positive + ', ' + value
            elif key == 'negative':
                m_negative = m_negative + ', ' + value

        cur += "--{key} \"{value}\" ".format(key='prompt', value= m_positive)
        cur += "--{key} \"{value}\" ".format(key='negative_prompt', value= m_negative)

        if parsed_param == False:
            params = parse_param("")
        cur += params
        cur += '--outpath_samples \"{} \"'.format(str(os.path.join(OUTPATH_SAMPLES, str(tag), '')))
        cur += ' --outpath_grids \"{} \"'.format(str(os.path.join(OUTPATH_GRIDS, str(tag), '')))
        return cur 
    else:
        for item in rawDict.items():
            key = item[0]
            ret = parse_yaml_dict(rawDict[key], tag if key=='' else key, avatar_prompt, avatar_name, default_negative)
            if len(ret) != 0:
                if tag not in EXCLUDED_TAGS:
                    OUTPUTS_DICT.append({'name': key,
                                    'prompt': item[1]['value'] if 'value' in item[1].keys() else '',
                                    'negative_prompt': item[1]['negative'] if 'negative' in item[1].keys() else ''})
                if tag in OUTPUTS.keys():
                    OUTPUTS[tag].append(ret)
                else:
                    OUTPUTS[tag] = [ret]
        return ""


def rename_preview(avatar_name):
    if avatar_name == '':
        print("Please select avatar name first.")
        return
    root = OUTPATH_SAMPLES
    for folder in os.listdir(root):
        files = os.listdir(os.path.join(root, folder))
        if 'Not-available.png' in files:
            print('Skip '+ folder + ' not available.')
            continue
        if avatar_name + '.png' in files:
            continue
        for each_avatar in avatar_names:
            if each_avatar + '.png' in files:
                files.remove(each_avatar + '.png')
        if len(files) == 1:
            os.rename(os.path.join(root, folder, files[0]), os.path.join(root, folder, avatar_name + '.png'))
        else:
            print('There are 0 or more than 1 files in ' + folder)

def load_prompt_file(file):
    if (file is None):
        lines = []
    else:
        lines = [x.strip() for x in file.decode('utf8', errors='ignore').split("\n")]

    return None, "\n".join(lines), gr.update(lines=7)

def copy_from_prompt_app():
    return []

def open_folder(f):
    if not os.path.exists(f):
        print(f'Folder "{f}" does not exist. After you create an image, the folder will be created.')
        return
    elif not os.path.isdir(f):
        print(f"""
WARNING
An open_folder request was made with an argument that is not a folder.
This could be an error or a malicious attempt to run code on your computer.
Requested path was: {f}
""", file=sys.stderr)
        return

    if not shared.cmd_opts.hide_ui_dir_config:
        path = os.path.normpath(f)
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            sp.Popen(["open", path])
        else:
            sp.Popen(["xdg-open", path])

class PromptStyle(typing.NamedTuple):
    name: str
    prompt: str
    negative_prompt: str

def save_styles() -> None:
    if len(OUTPUTS.keys()) == 0:
        return
    path = os.path.join('./', 'styles.csv')
    # Write to temporary file first, so we don't nuke the file if something goes wrong
    fd, temp_path = tempfile.mkstemp(".csv")
    with os.fdopen(fd, "a", encoding="utf-8-sig", newline='') as file:
        # _fields is actually part of the public API: typing.NamedTuple is a replacement for collections.NamedTuple,
        # and collections.NamedTuple has explicit documentation for accessing _fields. Same goes for _asdict()
        writer = csv.DictWriter(file, fieldnames=PromptStyle._fields)
        writer.writeheader()
        for row in OUTPUTS_DICT:
            writer.writerow({'name': row['name'], 'prompt': row['prompt'], 'negative_prompt': row['negative_prompt']})
        # writer.writerows(style._asdict() for k,     style in self.styles.items())

    # Always keep a backup file around
    if os.path.exists(path) and not os.path.exists(path + ".bak"):
        shutil.move(path, path + ".bak")
    shutil.move(temp_path, path)

def load_prompt(file, default_positive, default_negative, dropdown, skip_exist):
    global SKIP_EXISTS
    SKIP_EXISTS = skip_exist
    if dropdown == '' or file is None:
        return
    rawDict = yaml.load(file, Loader = yaml.BaseLoader)
    if len(default_positive) != 0:
        default_positive = avatar_prompts[avatar_names.index(dropdown)] + ', ' + default_positive
    else:
        default_positive = avatar_prompts[avatar_names.index(dropdown)]
    if len(default_negative) != 0:
        default_negative = avatar_negatives[avatar_names.index(dropdown)] + ', ' + default_negative
    else:
        default_negative = avatar_negatives[avatar_names.index(dropdown)]
    parse_yaml_dict(rawDict, "", default_positive, dropdown, default_negative)
    prompt_txt = ""
    keys = list(filter(lambda x: x not in EXCLUDED_TAGS, OUTPUTS.keys()))
    for style in keys:
        for each_line in OUTPUTS[style]:
            prompt_txt += each_line + '\n'
    return [prompt_txt, gr.Row.update(visible=True)]

def load_avartar(avatar_dict):
    avatars = yaml.load(avatar_dict, yaml.BaseLoader)

    for name, prompt in avatars.items():
        avatar_names.append(name)
        if 'value' in prompt.keys():
            avatar_prompts.append(prompt['value'])
        else:
            avatar_prompts.append('')
        if 'negative' in prompt.keys():
            avatar_negatives.append(prompt['negative'])
        else:
            avatar_negatives.append('')
    return [gr.Dropdown.update(choices=avatar_names, value=avatar_names[0]), gr.Column.update(visible=True),  gr.Group.update(visible=True)] 

def scan_outputs(avatar_name):
    if avatar_name is None or len(avatar_name) == 0:
        print("Please select avatar name first.")
        return
    root = OUTPATH_SAMPLES
    global qc_dict
    qc_dict = {}

    if not os.path.exists(root):
        os.mkdir(root)

    for folder in os.listdir(root):
        if os.path.isdir(os.path.join(root, folder)) == False:
            continue
            
        files = os.listdir(os.path.join(root, folder))
        if 'Not-available.png' in files:
            print('Skip '+ folder + ' not available.')
            continue
        if avatar_name + '.png' in files:
            continue
        for each_avatar in avatar_names:
            if each_avatar + '.png' in files:
                files.remove(each_avatar + '.png')
        if len(files) == 0:
            continue
        # print("Insert file:")
        for file in files:
            print(os.path.join(root, folder, file))
        qc_dict[folder] = [os.path.join(root, folder, file) for file in files]

    if len(qc_dict.keys()) == 0:
        return gr.Dropdown.update(choices=[])
    return gr.Dropdown.update(choices=list(qc_dict.keys()), value=list(qc_dict.keys())[0])

def update_gallery(dropdown, avatar):
    root = OUTPATH_SAMPLES
    global trg_img, current_folder
    current_folder = os.path.join(root, dropdown)
    trg_img = os.path.join(root, dropdown, avatar + '.png')
    # print("Detected folders:")
    # print(qc_dict)
    # print("Selected folder:")
    # print(dropdown)
    # print("Detected files:")
    # print(qc_dict[dropdown])
    return qc_dict[dropdown]

def clean_select_picture(filename):
    if current_folder == '':
        print("Please select qc tag.")
        return
    for file in os.listdir(current_folder):
        is_avatar = False
        for each_avatar in avatar_names:
            if each_avatar + '.png' == file:
                is_avatar = True
                break
        if '-' in file and file.split('.')[0] in filename:
            print("rename", os.path.join(current_folder, file), trg_img)
            os.rename(os.path.join(current_folder, file), trg_img)
        elif is_avatar == False:
            print(file, "delete", os.path.join(current_folder, file))
            os.remove(os.path.join(current_folder, file))

def image_url(filedata):
    if type(filedata) == list:
        if len(filedata) == 0:
            return None
        filedata = filedata[0]
        # for item in filedata:
        #     if filedata["is_file"]:
        #         filedata = item
        #         filename = filedata["name"]
        #         tempdir = os.path.normpath(tempfile.gettempdir())
        #         normfn = os.path.normpath(filename)
        #         assert normfn.startswith(tempdir), 'trying to open image file not in temporary directory'

        #         image = Image.open(filename)
        #         clean_select_picture(os.path.basename(filename))
                # return Image.open(filename)
    if type(filedata) == dict and filedata["is_file"]:
        filename = filedata["name"]
        tempdir = os.path.normpath(tempfile.gettempdir())
        normfn = os.path.normpath(filename)
        assert normfn.startswith(tempdir), 'trying to open image file not in temporary directory'

        image = Image.open(filename)
        clean_select_picture(os.path.basename(filename))
        return Image.open(filename)

    elif type(filedata) == dict:
        print(filedata)
        print("Dict is not file.")
        return 



    elif  type(filedata) != dict and filedata.startswith("data:image/png;base64,"):
        filedata = filedata[len("data:image/png;base64,"):]

        filedata = base64.decodebytes(filedata.encode('utf-8'))
        image = Image.open(io.BytesIO(filedata))
        return image
    return None
        

def dropdown_change():
    global OUTPUTS, OUTPUTS_DICT
    default_negative = []
    OUTPUTS = {}
    OUTPUTS_DICT = []
    return [ gr.File.update(value=None), gr.Textbox.update(value=None)]   
                    

class Script(scripts.Script):
    def title(self):
        return "Prompt gallery"

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Column():
                label_avatar = gr.Label("Upload avatars config")
                avatar_dict = gr.File(label="Upload avatar prompt inputs", type='binary')
        
        # copy_from_app_button = gr.Button("Copy From Prompt Preview")

        with gr.Group():
            with gr.Column(visible=False) as avatar_col:
                label_presets = gr.Label("Presets")
                dropdown = gr.Dropdown(label="Choose avatar", choices=[""], value="", type="value", elem_id="dropdown")
                dropdown.save_to_config = True
                with gr.Row():
                    checkbox_iterate = gr.Checkbox(label="Iterate seed every line", value=False)
                    skip_exist = gr.Checkbox(value=True, label="skip exist")
                default_negative = gr.Textbox(label="default_negative", lines=1)
                default_positive = gr.Textbox(label="default_positive", lines=1)
                prompt_dict = gr.File(label="Upload prompt dictionary", type='binary')
                with gr.Row(visible = False) as save_prompts:
                    open_button = gr.Button("Open outputs directory")
                    export_button = gr.Button("Export to WebUI style")
            prompt_display = gr.Textbox(label="List of prompt inputs", lines=1)

        
        prompt_dict.change(fn=load_prompt, inputs=[prompt_dict, default_positive, default_negative, dropdown, skip_exist], outputs=[prompt_display, save_prompts])
        open_button.click(fn=lambda: open_folder(OUTPATH_SAMPLES), inputs=[], outputs=[])
        export_button.click(fn=save_styles, inputs=[], outputs=[])

        with gr.Group(visible=False) as qc_widgets:
            label_preview = gr.Label("QC preview")
            with gr.Row():
                qc_refresh = gr.Button("QC scan")
                preview_dropdown = gr.Dropdown(label="Select prompts", choices=[""], value="", type="value", elem_id="dropdown")
            #review_gallery = gr.Gallery(label='Output', show_label=False, elem_id=f"preview_gallery").style(grid=4)
            preview_gallery = gr.Gallery(label="Output", show_label=False, elem_id="preview_gallery", columns=4)
            qc_refresh.click(fn=scan_outputs, inputs=[dropdown], outputs=preview_dropdown)
            with gr.Row():
                qc_show = gr.Button(f"Show pics")
                qc_select = gr.Button(f"Select")
                rename_button = gr.Button("Auto rename")
            # selected_img = gr.Image(label="Selected",show_label=False, source="upload", interactive=True, type="pil").style(height=480)
            selected_img = gr.Image(
                label="Selected",
                show_label=False,
                source="upload",
                interactive=True,
                type="pil",
                height=480,
            )
        qc_show.click(fn=update_gallery, inputs=[preview_dropdown, dropdown], outputs=preview_gallery)
        qc_select.click(
                    fn=lambda x: image_url(x),
                    _js="extract_image_from_gallery",
                    inputs=[preview_gallery],
                    outputs=[selected_img],
        )
            # qc_select.click(fn=select_picture, inputs=[dropdown, preview_dropdown, preview_gallery], outputs=[])
        dropdown.change(fn=dropdown_change, inputs=[], outputs=[prompt_dict, prompt_display])
        rename_button.click(fn=rename_preview, inputs=[dropdown], outputs=[])
            # qc_select.click(fn=scan_outputs, inputs=[], outputs=[preview_dropdown])

        avatar_dict.change(fn=load_avartar, inputs=[avatar_dict], outputs=[dropdown, avatar_col, qc_widgets])
        return [checkbox_iterate, avatar_dict, prompt_dict, default_negative, default_positive, dropdown, prompt_display, rename_button, label_avatar, open_button, export_button, skip_exist, label_presets, label_preview, preview_dropdown, preview_gallery, qc_select, qc_refresh, qc_show, selected_img]

    def run(self, p, checkbox_iterate, avatar_dict, prompt_dict, default_negative, default_positive, dropdown, prompt_display, rename_button, label_avatar, open_button, export_button, skip_exist, label_presets, label_preview, preview_dropdown, preview_gallery, qc_select, qc_refresh, qc_show, selected_img):
        global pg_templates
        backup = copy.deepcopy(shared.opts)

        shared.opts.data.update(pg_templates)
        lines = [x.strip() for x in prompt_display.splitlines()]
        lines = [x for x in lines if len(x) > 0]

        p.do_not_save_grid = True

        job_count = 0
        jobs = []

        for line in lines:
            if "--" in line:
                try:
                    args = cmdargs(line)
                except Exception:
                    print(f"Error parsing line [line] as commandline:", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
                    args = {"prompt": line}
            else:
                args = {"prompt": line}

            n_iter = args.get("n_iter", 1)
            if n_iter != 1:
                job_count += n_iter
            else:
                job_count += 1

            jobs.append(args)

        # print(f"Will process {len(lines)} lines in {job_count} jobs.")
        if (checkbox_iterate and p.seed == -1):
            p.seed = int(random.randrange(4294967294))

        state.job_count = job_count

        images = []
        for n, args in enumerate(jobs):
            state.job = f"{state.job_no + 1} out of {state.job_count}"

            copy_p = copy.copy(p)
            for k, v in args.items():
                setattr(copy_p, k, v)

            proc = process_images(copy_p)
            images += proc.images
            
            if (checkbox_iterate):
                p.seed = p.seed + (p.batch_size * p.n_iter)

        OUTPUTS = {}
        rawDict = {}
        
        shared.opts = backup
        return Processed(p, images, p.seed, "")
