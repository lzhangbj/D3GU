import torch
import torch.nn as nn

from actmtda.utils.registry import HEAD


def init_weights(m):
	classname = m.__class__.__name__
	if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
		nn.init.kaiming_uniform_(m.weight)
		nn.init.zeros_(m.bias)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight, 1.0, 0.02)
		nn.init.zeros_(m.bias)
	elif classname.find('Linear') != -1:
		nn.init.xavier_normal_(m.weight)
		nn.init.zeros_(m.bias)


@HEAD.register("multi-fc")
class MultiFCHead(nn.Module):
	def __init__(self, cfg):
		super(MultiFCHead, self).__init__()
		output_dim = cfg.OUTPUT_DIM
		hidden_dims = cfg.HIDDEN_DIMS
		act = cfg.ACTIVATION
		dropout = cfg.DROPOUT
		
		self.n_head = cfg.NUM_HEAD
		
		for i in range(self.n_head):
			linear_fc_list = []
			if act == 'relu':
				act_module = nn.ReLU
			elif act == 'leaky-relu':
				act_module = nn.LeakyReLU
			else:
				raise Exception()
			
			input_dim = cfg.INPUT_DIM
			for hidden_dim in hidden_dims:
				linear_fc_list.append(nn.Linear(input_dim, hidden_dim))
				linear_fc_list.append(act_module())
				linear_fc_list.append(nn.Dropout(dropout))
				input_dim = hidden_dim
			
			linear_fc_list.append(nn.Linear(input_dim, output_dim))
			fcs = nn.ModuleList(linear_fc_list)
			setattr(self, f"fc_{i}", fcs)
		self.apply(init_weights)
	
	def forward(self, x):
		feats = []
		for i in range(self.n_head):
			feat = x
			for fc in getattr(self, f"fc_{i}"):
				feat = fc(feat)
			feats.append(feat)
		feats = torch.cat(feats, dim=1)
		return feats
