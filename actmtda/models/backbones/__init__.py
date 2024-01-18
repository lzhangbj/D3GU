from .resnet import ResNet18, ResNet34, ResNet50, ResNet101

from actmtda.utils.registry import BACKBONE


def build_backbone(backbone_cfg):
	assert backbone_cfg.TYPE in BACKBONE, \
		"cfg.MODEL.BACKBONE.TYPE: {} are not registered in registry".format(
			backbone_cfg.TYPE
		)
	model = BACKBONE[backbone_cfg.TYPE](backbone_cfg)
	return model


__all__ = [
	"ResNet18", "ResNet34", "ResNet50", "ResNet101"
]