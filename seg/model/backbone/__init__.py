from seg.model.backbone.resnet import ResNet

backbones = {"resnet": ResNet}

def make_backbone(cfg):
    assert cfg.type in backbones, "Unknow backbone type {}".format(cfg.type)

    return backbones[cfg.type](**cfg)