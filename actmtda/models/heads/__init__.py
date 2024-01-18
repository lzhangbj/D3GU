from .fc import FCHead
from .multi_fc import MultiFCHead
from .cos_fc import CosineFCHead
from .normw_fc import NormWeightFCHead

from actmtda.utils.registry import HEAD


def build_head(head_cfg):
	assert head_cfg.TYPE in HEAD, \
		"cfg.MODEL.HEAD.TYPE: {} are not registered in registry".format(
			head_cfg.TYPE
		)
	model = HEAD[head_cfg.TYPE](head_cfg)
	return model

__all__ = [
	"FCHead", "MultiFCHead", "cos_fc", "NormWeightFCHead"
]