import os
import shutil
import json

from typing import Any
from dataclasses import dataclass
from dataclasses import asdict, fields, is_dataclass


from .tokenizer.config import TokenizerConfiguration
from .models.encoder   import encoder_config_by_type

@dataclass
class NBFConfiguration:

    tokenizer_config       : TokenizerConfiguration

    embedding_type         : str = "mean_pool"
    nbf_head_type          : str = "joint"

    encoder_model_type     : str = "transformer"
    encoder_model_config   : Any = None

    max_sequence_length    : int = 512
    vocabulary_size        : int = -1
    target_vocabulary_size : int = -1

    embedding_size : int = 512
    hidden_size    : int = 512
    pad_token_id   : int = 0

    def __post_init__(self):
        if self.encoder_model_config is None:
            self.encoder_model_config = encoder_config_by_type(self.encoder_model_type)()
        elif isinstance(self.encoder_model_config, dict):
            self.encoder_model_config = encoder_config_by_type(self.encoder_model_type)(
                **self.encoder_model_config
            )


# Config serialization ----------------------------------------------------------------

def save_config_to_json(config, target_dir):
    config_path = os.path.join(target_dir, 'config.json')
    serialized_config = _serialize_config(config, target_dir)

    with open(config_path, 'w') as o:
        json.dump(serialized_config, o, indent=4)


def _handle_file_paths(file_path, path_to_config):
    target_dir = path_to_config
    file_name = os.path.basename(file_path)
    new_path = os.path.join(".", file_name)

    shutil.copyfile(file_path, os.path.join(target_dir, new_path))
    return new_path


def _serialize_to_dict(path_to_config):

    def to_dict_fn(elements):
        
        def _handle_entry(entry):
            if isinstance(entry[1], str) and os.path.isfile(entry[1]):
                return (entry[0], _handle_file_paths(entry[1], path_to_config))
            return entry

        return dict(map(_handle_entry, elements))

    return to_dict_fn


def _serialize_config(config, path_to_config):
    return asdict(config, dict_factory=_serialize_to_dict(path_to_config))

# Config deserialization ----------------------------------------------------------------

def load_config_from_json(config_path):
    with open(config_path, "r") as f:
        config_json = json.load(f)

    base_dir = os.path.dirname(config_path)
    
    return _build_config_from_json(config_json, base_dir)


def _parse_config_with_class(clazz, json_dict, base_dir):
    kwargs = {}

    for field in fields(clazz):
        name = field.name
        if name in json_dict:
            json_entry = json_dict[name]
            if json_entry is None: continue
            field_type = field.type
            if is_dataclass(field_type):
                kwargs[name] = _parse_config_with_class(field_type, json_entry, base_dir)
            elif field_type is str and json_entry.startswith("."):
                kwargs[name] = os.path.join(base_dir, json_entry)
            else:
                kwargs[name] = json_entry

    result = clazz(**kwargs)

    if hasattr(result, "deserialize_init"):
        result.deserialize_init()

    return result


def _build_config_from_json(json_dict, base_dir):
    root_config_class = NBFConfiguration
    return _parse_config_with_class(root_config_class, json_dict, base_dir)

