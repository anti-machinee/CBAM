from .modules.resnet.resnet50 import ResNet50


def get_model(name, **kwargs):
    if name == "resnet50":
        return ResNet50(num_classes=10)
    else:
        raise NameError(f"Model name {name} is invalid")
