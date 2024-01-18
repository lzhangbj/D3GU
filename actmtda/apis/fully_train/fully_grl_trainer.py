import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

import os
import random
import json
import numpy as np
from tqdm import tqdm
import time
import datetime
import logging

import wandb

from actmtda.models.recognizers import build_recognizer
from actmtda.models.heads import build_head
from actmtda.samplers import build_sampler
from actmtda.datasets import build_dataset, build_dataloader
from actmtda.losses import CosineAnnealingLR_with_Restart, build_loss
from actmtda.utils import AverageMeter, ReverseLayerF


def seed_everything(seed=888):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def build_optimizer(model, discriminator, cfg):
	optimizer = cfg.SOLVER.OPTIMIZER
	lr = cfg.SOLVER.LR
	backbone_lr_rescale = cfg.SOLVER.BACKBONE_LR_RESCALE
	weight_decay = cfg.SOLVER.WEIGHT_DECAY
	
	param = [
		{'params': model.backbone.parameters(), "lr": lr * backbone_lr_rescale},
		{'params': model.head.parameters(), "lr": lr},
		{'params': discriminator.parameters(), "lr": lr}
	]
	if optimizer == 'sgd':
		optimizer = optim.SGD(param, momentum=0.9, weight_decay=weight_decay, nesterov=True)
	elif optimizer == 'adam':
		optimizer = optim.Adam(param, betas=(0.9, 0.999), weight_decay=weight_decay)
	else:
		raise Exception()
	
	return optimizer


class FullyGrlTrainer:
	def __init__(self, cfg):
		self.cfg = cfg
		self.work_dir = cfg.WORK_DIR
		# init seed
		seed_everything(cfg.RANDOM_SEED)
		
		logging.basicConfig(format='[%(asctime)s-%(levelname)s]: %(message)s',
		                    level=logging.INFO,
		                    handlers=[
			                    logging.StreamHandler(),
			                    logging.FileHandler(os.path.join(cfg.WORK_DIR, 'train.log'), "a"),
		                    ])
		self.logger = logging.getLogger("actmtda.trainer")
		
		self.net = build_recognizer(cfg.MODEL)
		if os.path.exists(cfg.MODEL.LOAD_FROM):
			self.load_pretrained(cfg.MODEL.LOAD_FROM)
		self.net.cuda()
		
		self.discriminator = build_head(cfg.MODEL.DISCRIMINATOR).cuda()
		self.merge_target = cfg.MODEL.DISCRIMINATOR.MERGE_TARGET
		if self.merge_target: assert cfg.MODEL.DISCRIMINATOR.OUTPUT_DIM == 2
		
		# active settings
		# target val dataset
		target_val_dataset = build_dataset(self.cfg.DATASET, self.cfg.DATASET.TARGET.DOMAIN_NAMES, domain_label_start=1,
											split='test', is_train=False)
		self.target_val_dataloader = build_dataloader(target_val_dataset,
													self.cfg.VAL.BATCH_SIZE,
													self.cfg.VAL.NUM_WORKER,
													is_train=False)
		
		# create sampler
		source_dataset = build_dataset(self.cfg.DATASET, self.cfg.DATASET.SOURCE.DOMAIN_NAMES, domain_label_start=0,
		                               split=None, is_train=True)
		self.source_dataloader = build_dataloader(source_dataset,
													self.cfg.TRAIN.BATCH_SIZE,
													self.cfg.TRAIN.NUM_WORKER//2,
													is_train=True)
		target_train_dataset = build_dataset(self.cfg.DATASET, self.cfg.DATASET.TARGET.DOMAIN_NAMES,
		                                     domain_label_start=1, split='train', is_train=True)
		self.target_train_dataloader = build_dataloader(target_train_dataset,
													self.cfg.TRAIN.BATCH_SIZE//4,
													self.cfg.TRAIN.NUM_WORKER//2,
													is_train=True)
		
		self.loss_fn = build_loss(cfg.LOSS)
		self.dom_loss_weight = cfg.LOSS.DOMAIN_DISC_WEIGHT
	
	def run(self):
		
		# initialize model and optimizer
		optimizer = build_optimizer(self.net, self.discriminator, self.cfg)
		
		n_iter_per_epoch = len(self.source_dataloader)
		expect_iter = self.cfg.TRAIN.NUM_EPOCH * n_iter_per_epoch
		
		# initialize scheduler
		if self.cfg.SOLVER.SCHEDULER == "CosineAnnealingLR_with_Restart":
			scheduler = CosineAnnealingLR_with_Restart(
				optimizer,
				T_max=self.cfg.SOLVER.COSINEANNEALINGLR.T_MAX * n_iter_per_epoch,
				T_mult=self.cfg.SOLVER.COSINEANNEALINGLR.T_MULT,
				eta_min=self.cfg.SOLVER.LR * 0.001
			)
		elif self.cfg.SOLVER.SCHEDULER == "LambdaLR":
			lr_lambda = lambda iter: (1 - (iter / expect_iter)) ** 0.9
			scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
		else:
			raise Exception()
		
		dataset_name = self.cfg.DATASET.DATASET_NAME
		source_domain_name = self.cfg.DATASET.SOURCE.DOMAIN_NAMES[0]
		target_domain_names = self.cfg.DATASET.TARGET.DOMAIN_NAMES
		self.logger.info(f"####### DANN supervised training on {dataset_name} {len(self.source_dataloader.dataset)} source({source_domain_name}) and {len(self.target_train_dataloader.dataset)} target data")
		
		max_target_acc = 0.
		max_target_accs = [0. for _ in range(len(target_domain_names))]
		max_target_acc_epoch = 0
		iter_report_start = time.time()
		
		epoch = 1
		iter_cnt = 0
		
		interval_loss = AverageMeter()
		interval_cls_loss = AverageMeter()
		interval_dom_loss = AverageMeter()
		
		source_iter = iter(self.source_dataloader)
		target_iter = iter(self.target_train_dataloader)
		while iter_cnt < expect_iter:
			# training
			self.net.train()
			self.discriminator.train()
			
			try:
				source_batch = next(source_iter)
			except StopIteration:
				epoch += 1
				assert iter_cnt % n_iter_per_epoch == 0, [iter_cnt, n_iter_per_epoch]
				source_iter = iter(self.source_dataloader)
				source_batch = next(source_iter)
			source_images = source_batch['image'].cuda()
			source_labels = source_batch['target'].cuda()
			source_domains = source_batch['domain'].cuda()
			B_source = source_images.size(0)
			
			try:
				target_batch = next(target_iter)
			except StopIteration:
				target_iter = iter(self.target_train_dataloader)
				target_batch = next(target_iter)
			target_images = target_batch['image'].cuda()
			target_labels = target_batch['target'].cuda()
			target_domains = target_batch['domain'].cuda()
			B_target = target_images.size(0)
			
			if self.merge_target:
				target_domains = torch.ones_like(target_domains)
			
			iter_cnt += 1
			
			images = torch.cat([source_images, target_images])
			domains = torch.cat([source_domains, target_domains])
			
			cls_logits, feats = self.net(images)
			labeled_cls_logits = cls_logits[:B_source+B_target]
			labeled_labels = torch.cat([source_labels, target_labels])
			cls_loss = self.loss_fn(labeled_cls_logits, labeled_labels)
			
			p = float(iter_cnt) / expect_iter
			alpha = 2. / (1. + np.exp(-10 * p)) - 1
			rev_feats = ReverseLayerF.apply(feats, alpha)
			dom_logits = self.discriminator(rev_feats)
			dom_loss = self.dom_loss_weight * self.loss_fn(dom_logits, domains)
			
			loss = cls_loss + dom_loss
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			interval_loss.update(loss.detach().cpu().item(), B_source)
			interval_cls_loss.update(cls_loss.detach().cpu().item(), B_source)
			interval_dom_loss.update(dom_loss.detach().cpu().item(), B_source)
			
			if iter_cnt % self.cfg.TRAIN.ITER_REPORT == 0:
				# eta
				iter_report_time = time.time() - iter_report_start
				eta = str(datetime.timedelta(seconds=int(iter_report_time * (expect_iter - iter_cnt) / iter_cnt))).split(".")[0]
				
				self.logger.info(
					'ETA:{}, Epoch:{}/{}, iter:{}/{}, lr:{:.5f}, loss:{:.4f}, cls-loss:{:.4f}, dom-loss:{:.4f}'.format(
						eta,
						epoch, self.cfg.TRAIN.NUM_EPOCH,
						iter_cnt, expect_iter,
						optimizer.param_groups[-1]['lr'],
						interval_loss.avg,
						interval_cls_loss.avg,
						interval_dom_loss.avg))
				interval_loss.reset()
				interval_cls_loss.reset()
				interval_dom_loss.reset()
			
			wandb.log({
				f"train/loss": loss.detach().cpu().item(),
				f"train/cls-loss": cls_loss.detach().cpu().item(),
				f"train/dom-loss": dom_loss.detach().cpu().item(),
				f"train/lr": optimizer.param_groups[-1]['lr'],
				'train_iter': iter_cnt
			})
			scheduler.step()
			
			# val
			if iter_cnt % (self.cfg.TRAIN.VAL_EPOCH * n_iter_per_epoch) == 0:
				self.net.eval()
				
				target_acc_meters = [AverageMeter() for _ in range(len(target_domain_names))]
				with torch.no_grad():
					for batch_data in self.target_val_dataloader:
						images = batch_data['image'].cuda()
						labels = batch_data['target'].long()
						domains = batch_data['domain'].long()
						
						logits, feats = self.net(images)
						preds = torch.argmax(logits, dim=1).detach().cpu()
						acc_mask = (preds == labels).float()
						for target_domain in range(len(target_domain_names)):
							indomain_idx = torch.nonzero(domains == target_domain + 1, as_tuple=True)[0]
							domain_acc = acc_mask[indomain_idx].mean().item()
							if len(indomain_idx):
								target_acc_meters[target_domain].update(domain_acc, len(indomain_idx))
				target_accs = [target_acc_meter.avg for target_acc_meter in target_acc_meters]
				target_mean_acc = np.mean(target_accs)
				if target_mean_acc > max_target_acc:
					max_target_acc = target_mean_acc
					max_target_accs = target_accs
					max_target_acc_epoch = epoch
					target_best_save_path = os.path.join(self.work_dir, "ckpt", "best_target_model.pth")
					save_dict = {
						"net": self.net.state_dict(),
						"target_mean_acc": max_target_acc,
						"epoch": epoch}
					for target_domain in range(len(target_domain_names)):
						save_dict[f"{target_domain_names[target_domain]}_acc"] = target_accs[target_domain]
					torch.save(save_dict, target_best_save_path)
				
				# save model
				save_path = os.path.join(self.work_dir, "ckpt", f"epoch_{epoch}.pth")
				save_dict = {
					"net": self.net.state_dict(),
					"target_mean_acc": target_mean_acc,
					"epoch": epoch
				}
				for target_domain in range(len(target_domain_names)):
					save_dict[f"{target_domain_names[target_domain]}_acc"] = target_accs[target_domain]
				torch.save(save_dict, save_path)
				
				log_info = 'VAL/Best Epoch:{}/{}, tgt acc: {:.4f}/{:.4f}'.format(
					epoch, max_target_acc_epoch,
					target_mean_acc, max_target_acc
				)
				for target_domain in range(len(target_domain_names)):
					log_info += "\t{}: {:.4f}/{:.4f}".format(target_domain_names[target_domain],
					                                         target_accs[target_domain], max_target_accs[target_domain])
				self.logger.info(log_info)
				
				wandb_log_dict = {
					'epoch': epoch,
					f'val/target_acc': target_mean_acc,
					f'val/target_best_acc': max_target_acc
				}
				for target_domain in range(len(target_domain_names)):
					wandb_log_dict[f"val/target_{target_domain_names[target_domain]}_acc"] = target_accs[target_domain]
					wandb_log_dict[f"val/target_{target_domain_names[target_domain]}_best_acc"] = max_target_accs[target_domain]
				
				wandb.log(wandb_log_dict)
		
		self.logger.info('######### End! ##########')



