import torch
import torch.nn as nn

from actmtda.models.backbones import build_backbone
from actmtda.models.necks import build_neck
from actmtda.models.heads import build_head
from actmtda.utils.registry import RECOGNIZER


@RECOGNIZER.register("general_recognizer")
class GeneralRecognizer(nn.Module):
	def __init__(self, cfg):
		super(GeneralRecognizer, self).__init__()
		self.backbone = build_backbone(cfg.BACKBONE)
		if cfg.NECK.TYPE:
			self.neck = build_neck(cfg.NECK)
		self.head = build_head(cfg.HEAD)
	
	def forward(self, x):
		feat = self.backbone(x)
		if hasattr(self, 'neck'):
			neck_feat = self.neck(feat)
			logits = self.head(neck_feat)
			return logits, neck_feat
		logits = self.head(feat)
		return logits, feat
	
	def encode_feat(self, x):
		feat = self.backbone(x)
		if hasattr(self, 'neck'):
			neck_feat = self.neck(feat)
			return neck_feat
		return feat
	
	def forward_classifier(self, feat, is_input_neck=False):
		if not is_input_neck and hasattr(self, 'neck'):
			feat = self.neck(feat)
		logits = self.head(feat)
		return logits