from .vit import ViT


def create_model(args):
    return ViT(args)
