from .EDM import EDM
from .unetC import UNetC
from .unetE import UNetE
from .unetG import UNetG


CLASSES = {
    cls.__name__: cls
    for cls in [EDM, UNetC, UNetE, UNetG]
}


def get_models_class(model_type, net_type):
    return CLASSES[model_type], CLASSES[net_type]
