from .losses import CrossEntropyLoss
from .cos_annealing_with_restart import CosineAnnealingLR_with_Restart

from actmtda.utils.registry import LOSS

def build_loss(loss_cfg):
	assert loss_cfg.TYPE in LOSS, \
		"cfg.LOSS.TYPE: {} are not registered in registry".format(
			loss_cfg.TYPE
		)
	loss = LOSS[loss_cfg.TYPE](loss_cfg)
	
	return loss


__all__ = [
	'CrossEntropyLoss', 'CosineAnnealingLR_with_Restart'
]