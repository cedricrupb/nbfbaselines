from .embeddings import embedding_from_config
from .heads      import nbf_head_from_config, convert_target_to_mask
from .encoder    import encoder_clazz_by_type
from .encoder    import encoder_config_from_model_size


def nbfmodel_from_config(config):
    embedding_table = embedding_from_config(config)
    prediction_head = nbf_head_from_config(config)

    encoder_model_config = config.encoder_model_config
    if hasattr(encoder_model_config, "copy_from_nbfconfig"):
        encoder_model_config.copy_from_nbfconfig(config)

    encoder_clazz = encoder_clazz_by_type(config.encoder_model_type)
    return encoder_clazz(
        encoder_model_config,
        embedding_table,
        prediction_head
    )