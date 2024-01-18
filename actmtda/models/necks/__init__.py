from .fc import FCNeck, IdentityNeck
from .normout_fc import NormoutFCNeck

from actmtda.utils.registry import NECK


def build_neck(neck_cfg):
	assert neck_cfg.TYPE in NECK, \
		"cfg.MODEL.NECK.TYPE: {} are not registered in registry".format(
			neck_cfg.TYPE
		)
	model = NECK[neck_cfg.TYPE](neck_cfg)
	return model

__all__ = [
	"FCNeck",
	"IdentityNeck",
	"NormoutFCNeck"
]