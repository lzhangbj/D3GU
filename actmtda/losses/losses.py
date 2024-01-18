import torch
import torch.nn as nn

from actmtda.utils.registry import LOSS

@LOSS.register("ce")
class CrossEntropyLoss():
	def __init__(self, cfg):
		self.loss = nn.CrossEntropyLoss(reduction='mean')
	
	def __call__(self, pred, gt):
		return self.loss(pred, gt)