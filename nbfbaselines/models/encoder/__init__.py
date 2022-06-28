

from .transformer import (
    TransformerConfig,
    TransformerNBF
)

from .transformer import config_from_model_size as transformer_config_from_model_size

MODEL_REGISTRY = {
    "transformer": (TransformerConfig, TransformerNBF)
}

def register_model(name, config_clazz, model_clazz):
    global MODEL_REGISTRY
    MODEL_REGISTRY[name] = (config_clazz, model_clazz)


def encoder_config_by_type(encoder_type):
    if encoder_type in MODEL_REGISTRY:
        return MODEL_REGISTRY[encoder_type][0]
    
    raise ValueError("Unknown encoder type %s" % encoder_type)


def encoder_clazz_by_type(encoder_type):
    if encoder_type in MODEL_REGISTRY:
        return MODEL_REGISTRY[encoder_type][1]
    
    raise ValueError("Unknown encoder type %s" % encoder_type)


def encoder_config_from_model_size(encoder_type, model_size):

    if encoder_type == "transformer":
        return transformer_config_from_model_size(model_size)

    raise ValueError("Unknown encoder type %s" % encoder_type)

