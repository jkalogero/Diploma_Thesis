from visdialch.encoders.KBGN import KBGN

def Encoder(model_config, *args):
    name_enc_map = {
        'kbgn': KBGN
    }
    return name_enc_map[model_config["encoder"]](model_config, *args)