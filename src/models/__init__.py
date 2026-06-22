from .fengyuan import FengYuan


MODEL_LIST = {
              'fengyuan': FengYuan,
              }


def get_model(model_name, configs):
    if model_name in MODEL_LIST.keys():
        model = MODEL_LIST.get(model_name)(**configs['model'])
    else:
        raise ValueError(f'Unknow model {model_name}!')

    return model

