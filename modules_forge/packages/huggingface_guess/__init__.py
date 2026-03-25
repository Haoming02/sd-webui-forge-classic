"""
Copyright (C) 2024 lllyasviel

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/
"""

from .detection import model_config_from_unet, unet_prefix_from_state_dict


def guess(state_dict):
    unet_key_prefix = unet_prefix_from_state_dict(state_dict)
    result = model_config_from_unet(state_dict, unet_key_prefix, use_base_if_no_match=False)
    result.unet_key_prefix = [unet_key_prefix]
    result.unet_config.pop("image_model", None)
    result.unet_config.pop("audio_model", None)
    return result


def guess_repo_name(state_dict):
    config = guess(state_dict)
    assert config is not None
    repo_id = config.huggingface_repo
    return repo_id
