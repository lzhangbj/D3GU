import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import os
import random
import numpy as np
import datetime
from tqdm import tqdm
import time
import logging

import wandb

from actmtda.models.recognizers import build_recognizer
from actmtda.models.heads import build_head
from actmtda.models.necks import build_neck
from actmtda.datasets import build_dataset, build_dataloader
from actmtda.losses import CosineAnnealingLR_with_Restart, build_loss
from actmtda.utils import AverageMeter, ReverseLayerF

from prettytable import PrettyTable


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
		{'params': model.neck.parameters(), "lr": lr},
		{'params': model.head.parameters(), "lr": lr},
		{'params': discriminator.parameters(), "lr": lr},
	]
	if optimizer == 'sgd':
		optimizer = optim.SGD(param, momentum=0.9, weight_decay=weight_decay)
	elif optimizer == 'adam':
		optimizer = optim.Adam(param, betas=(0.9, 0.999), weight_decay=weight_decay)
	else:
		raise Exception()
	
	return optimizer


class PretrainDANNBinTrainer:
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
		
		assert not self.cfg.TRAIN.TRAIN_TEACHER, "Only support single model training"
		
		seed_everything(cfg.RANDOM_SEED)
		self.student_net = build_recognizer(cfg.MODEL).cuda()
		seed_everything(cfg.RANDOM_SEED)
		self.student_discriminator = build_head(cfg.MODEL.DISCRIMINATOR).cuda()
		
		seed_everything(cfg.RANDOM_SEED)
		# source train dataset
		if self.cfg.DATASET.DATASET_NAME == 'domain-net':
			source_train_dataset = build_dataset(self.cfg.DATASET, self.cfg.DATASET.SOURCE.DOMAIN_NAMES,
			                                     domain_label_start=0, split='train', is_train=True)
		else:
			source_train_dataset = build_dataset(self.cfg.DATASET, self.cfg.DATASET.SOURCE.DOMAIN_NAMES,
			                                     domain_label_start=0, split=None, is_train=True)
		self.source_train_dataloader = build_dataloader(source_train_dataset,
		                                                self.cfg.TRAIN.BATCH_SIZE,
		                                                self.cfg.TRAIN.NUM_WORKER,
		                                                is_train=True)
		# target train datasets
		self.n_target = len(self.cfg.DATASET.TARGET.DOMAIN_NAMES)
		self.target_train_dataloaders = []
		for i, target_domain_name in enumerate(self.cfg.DATASET.TARGET.DOMAIN_NAMES):
			if self.cfg.DATASET.DATASET_NAME == 'domain-net':
				target_train_dataset = build_dataset(self.cfg.DATASET, [target_domain_name, ],
				                                     domain_label_start=i + 1, split='train', is_train=True)
			else:
				target_train_dataset = build_dataset(self.cfg.DATASET, [target_domain_name, ],
				                                     domain_label_start=i + 1, split=None, is_train=True)
			self.target_train_dataloaders.append(build_dataloader(target_train_dataset,
			                                                      self.cfg.TRAIN.TARGET.UNLABELED.BATCH_SIZE,
			                                                      self.cfg.TRAIN.TARGET.UNLABELED.NUM_WORKER,
			                                                      is_train=True))
		# source val dataset
		if self.cfg.DATASET.DATASET_NAME == 'domain-net':
			source_val_dataset = build_dataset(self.cfg.DATASET, self.cfg.DATASET.SOURCE.DOMAIN_NAMES,
			                                   domain_label_start=0, split='test', is_train=False)
		else:
			source_val_dataset = build_dataset(self.cfg.DATASET, self.cfg.DATASET.SOURCE.DOMAIN_NAMES,
			                                   domain_label_start=0, split=None, is_train=False)
		self.source_val_dataloader = build_dataloader(source_val_dataset,
		                                              self.cfg.VAL.BATCH_SIZE,
		                                              self.cfg.VAL.NUM_WORKER,
		                                              is_train=False)
		# target val dataset
		if self.cfg.DATASET.DATASET_NAME == 'domain-net':
			target_val_datasets = [
				build_dataset(self.cfg.DATASET, [target_domain_name, ], domain_label_start=target_domain_idx + 1,
				              split='test', is_train=False) for target_domain_idx, target_domain_name in
				enumerate(self.cfg.DATASET.TARGET.DOMAIN_NAMES)]
		else:
			target_val_datasets = [
				build_dataset(self.cfg.DATASET, [target_domain_name, ], domain_label_start=target_domain_idx + 1,
				              split=None, is_train=False) for target_domain_idx, target_domain_name in
				enumerate(self.cfg.DATASET.TARGET.DOMAIN_NAMES)]
		
		self.target_val_dataloaders = [build_dataloader(target_val_dataset,
		                                                self.cfg.VAL.BATCH_SIZE,
		                                                self.cfg.VAL.NUM_WORKER,
		                                                is_train=False) for target_val_dataset in target_val_datasets]
		
		self.loss_fn = build_loss(cfg.LOSS)
		self.dom_loss_weight = cfg.LOSS.DOMAIN_DISC_WEIGHT
		self.balance_da_loss = self.cfg.LOSS.BALANCE_DA_LOSS
	
	def run(self):
		n_target = len(self.cfg.DATASET.TARGET.DOMAIN_NAMES)
		
		student_optimizer = build_optimizer(self.student_net, self.student_discriminator, self.cfg)
		
		n_iter_per_epoch = len(self.source_train_dataloader)
		expect_iter = self.cfg.TRAIN.NUM_EPOCH * n_iter_per_epoch
		if self.cfg.SOLVER.SCHEDULER == "LambdaLR":
			lr_lambda = lambda iter: (1 - (iter / expect_iter)) ** 0.9
			student_scheduler = LambdaLR(student_optimizer, lr_lambda=lr_lambda)
		elif self.cfg.SOLVER.SCHEDULER == "InvLR":
			lr_lambda = lambda iter: (1 + (10 * iter / expect_iter)) ** (-0.75)
			student_scheduler = LambdaLR(student_optimizer, lr_lambda=lr_lambda)
		else:
			raise Exception()
		
		dataset_name = self.cfg.DATASET.DATASET_NAME
		source_domain_name = self.cfg.DATASET.SOURCE.DOMAIN_NAMES[0]
		target_domain_names = self.cfg.DATASET.TARGET.DOMAIN_NAMES
		self.logger.info(
			f"####### Random Ensembled DANN bin pretrain for dataset {dataset_name}-{source_domain_name} #########")
		
		student_max_mean_acc_epoch = 0
		student_max_mean_acc = 0.
		student_max_accs = np.zeros(n_target)
		
		iter_report_start = time.time()
		
		epoch = 1
		iter_cnt = 0
		
		interval_student_loss = AverageMeter()
		interval_student_cls_loss = AverageMeter()
		interval_student_discrim_loss = AverageMeter()
		
		source_train_iter = iter(self.source_train_dataloader)
		target_train_iters = [iter(target_train_dataloader) for target_train_dataloader in
		                      self.target_train_dataloaders]
		while iter_cnt < expect_iter:
			# training
			self.student_net.train()
			self.student_discriminator.train()
			
			try:
				source_batch = next(source_train_iter)
			except:
				assert iter_cnt % n_iter_per_epoch == 0, [iter_cnt, n_iter_per_epoch, epoch]
				epoch += 1
				source_train_iter = iter(self.source_train_dataloader)
				source_batch = next(source_train_iter)
			iter_cnt += 1
			source_images = source_batch['image'].cuda()
			source_labels = source_batch['target'].cuda()
			source_domains = torch.zeros_like(source_labels).long().cuda()
			B_source = source_images.size(0)
			
			concat_images_list = [source_images, ]
			concat_domains_list = [source_domains, ]
			for tgt_idx in range(n_target):
				try:
					target_batch = next(target_train_iters[tgt_idx])
				except:
					target_train_iters[tgt_idx] = iter(self.target_train_dataloaders[tgt_idx])
					target_batch = next(target_train_iters[tgt_idx])
				target_images = target_batch['image'].cuda()
				target_domains = target_batch['domain'].cuda()
				B_target = target_images.size(0)
				concat_images_list.append(target_images)
				# crucial part of V1 trainer
				concat_domains_list.append(torch.ones_like(target_domains))
			
			concat_images = torch.cat(concat_images_list)
			concat_domains = torch.cat(concat_domains_list)
			
			# domain discriminator coeff
			p = float(iter_cnt) / expect_iter
			alpha = 2. / (1. + np.exp(-10 * p)) - 1
			
			# first forward students to produce weights from discriminator
			student_cls_logits, student_concat_features = self.student_net(concat_images)
			student_src_cls_logits = student_cls_logits[:B_source]
			
			# update student discrim loss
			student_rev_concat_features = ReverseLayerF.apply(student_concat_features, alpha)
			student_concat_discrim_logits = self.student_discriminator(student_rev_concat_features)
			da_weight = torch.ones(2).float().cuda()
			if self.balance_da_loss:
				da_weight[1] /= n_target
				da_weight = da_weight.contiguous()
			student_discrim_loss = self.dom_loss_weight * F.cross_entropy(student_concat_discrim_logits, concat_domains, weight=da_weight)
			
			student_cls_loss = F.cross_entropy(student_src_cls_logits, source_labels)
			
			# update student model
			student_loss = student_cls_loss + student_discrim_loss
			
			student_optimizer.zero_grad()
			student_loss.backward()
			student_optimizer.step()
			student_scheduler.step()
			
			interval_student_loss.update(student_loss.detach().cpu().item(), B_source)
			interval_student_cls_loss.update(student_cls_loss.detach().cpu().item(), B_source)
			interval_student_discrim_loss.update(student_discrim_loss.detach().cpu().item(), B_source)
			
			wandb.log({
				f"train/stu_loss": student_loss.detach().cpu().item(),
				f"train/stu_cls-loss": student_cls_loss.detach().cpu().item(),
				f"train/stu_discrim-loss": student_discrim_loss.detach().cpu().item(),
				'train_iter': iter_cnt
			})
			
			if iter_cnt % self.cfg.TRAIN.ITER_REPORT == 0:
				# eta
				iter_report_time = time.time() - iter_report_start
				eta = \
				str(datetime.timedelta(seconds=int(iter_report_time * (expect_iter - iter_cnt) / iter_cnt))).split(".")[
					0]
				
				log_info = f"ETA:{eta}, Epoch:{epoch}/{self.cfg.TRAIN.NUM_EPOCH}, iter:{iter_cnt}/{expect_iter}, lr:{student_optimizer.param_groups[-1]['lr']:.5f}, "
				log_info += f"student-loss / cls-loss / discrim-loss: {interval_student_loss.avg:.4f} / {interval_student_cls_loss.avg:.4f} / {interval_student_discrim_loss.avg:.4f}"
				self.logger.info(log_info)
				
				interval_student_loss.reset()
				interval_student_cls_loss.reset()
				interval_student_discrim_loss.reset()
			
			# val
			if iter_cnt % (self.cfg.TRAIN.VAL_EPOCH * n_iter_per_epoch) == 0:
				self.student_net.eval()
				self.student_discriminator.eval()
				
				student_source_acc_meter = AverageMeter()
				student_target_acc_meters = [AverageMeter() for _ in range(n_target)]
				student_domain_prob_meters = [AverageMeter() for _ in range(n_target+1)]
				# compute average discriminator outputs
				with torch.no_grad():
					# loop each source for validation
					for batch_data in self.source_val_dataloader:
						images = batch_data['image'].cuda()
						labels = batch_data['target'].long()
						domains = batch_data['domain'].long().detach().cpu().numpy()
						B = len(images)
						
						# eval student acc
						logits, feats = self.student_net(images)
						preds = torch.argmax(logits, dim=1).detach().cpu()
						acc = (preds == labels).float().mean().item() * 100
						student_source_acc_meter.update(acc, B)
						# compute student discriminator probs
						student_source_probs = torch.softmax(self.student_discriminator(feats), dim=1).detach().cpu().numpy()
						student_source_probs = student_source_probs.mean(axis=0)
						student_domain_prob_meters[0].update(student_source_probs, B)
						
					# loop each target for validation
					for tgt_dataset_idx, target_val_dataloader in enumerate(self.target_val_dataloaders):
						for batch_data in target_val_dataloader:
							images = batch_data['image'].cuda()
							labels = batch_data['target'].long()
							domains = batch_data['domain'].long().detach().cpu().numpy()
							B = len(images)
							
							# eval student acc
							logits, feats = self.student_net(images)
							preds = torch.argmax(logits, dim=1).detach().cpu()
							acc = (preds == labels).float().mean().item() * 100
							student_target_acc_meters[tgt_dataset_idx].update(acc, B)
							# compute student discriminator probs
							student_dom_probs = torch.softmax(self.student_discriminator(feats), dim=1).detach().cpu().numpy()
							for dom_id in range(self.n_target+1):
								indom_idx = np.nonzero(domains==dom_id)[0]
								if len(indom_idx) == 0: continue
								student_indom_probs = student_dom_probs[indom_idx].mean(axis=0)
								student_domain_prob_meters[dom_id].update(student_indom_probs, len(indom_idx))
							
				student_source_acc = student_source_acc_meter.avg
				student_target_accs = np.zeros(n_target)
				student_dom_probs = np.zeros((n_target+1, 2))
				for target_idx in range(n_target):
					student_target_accs[target_idx] = student_target_acc_meters[target_idx].avg
				for dom_idx in range(n_target+1):
					student_dom_probs[dom_idx] = student_domain_prob_meters[dom_idx].avg
				student_mean_acc = np.mean(student_target_accs)
				
				save_dict = {
					"net": self.student_net.state_dict(),
					"discriminator": self.student_discriminator.state_dict(),
					"target_accs": student_target_accs,
					"epoch": epoch}
				
				# save model
				save_path = os.path.join(self.work_dir, "ckpt", f"last.pth")
				torch.save(save_dict, save_path)
				
				if student_mean_acc > student_max_mean_acc:
					student_max_mean_acc = student_mean_acc
					student_max_accs = student_target_accs
					student_max_mean_acc_epoch = epoch
					student_best_save_path = os.path.join(self.work_dir, "ckpt", "best_model.pth")
					torch.save(save_dict, student_best_save_path)
				
				DOMAIN_NAMES = self.cfg.DATASET.SOURCE.DOMAIN_NAMES + self.cfg.DATASET.TARGET.DOMAIN_NAMES
				
				ptable = PrettyTable()
				ptable.field_names = ["model", 'best-epoch'] + DOMAIN_NAMES + ['mean', ]
				ptable.add_row(['main', student_max_mean_acc_epoch] +
				               [f"{student_source_acc:.2f}", ] +
				               [f"{student_target_accs[tgt]:.2f}/{student_max_accs[tgt]:.2f}" for tgt in range(n_target)] +
				               [f"{student_mean_acc:.2f}/{student_max_mean_acc:.2f}", ])
				
				
				dom_ptable = PrettyTable()
				dom_ptable.field_names = ["model", "data domain", DOMAIN_NAMES[0], "-".join(DOMAIN_NAMES[1:])]
		
				# student
				for domain_idx in range(n_target + 1):
					row = [DOMAIN_NAMES[domain_idx], ] + [f"{prob:.2f}" for prob in student_dom_probs[domain_idx]]
					if domain_idx == 0: row = ["main", ] + row
					else: row = ["", ] + row
					dom_ptable.add_row(row)
				log_info = f'VAL Epoch {epoch}\n{ptable}\n{dom_ptable}'
				
				# eta
				iter_report_time = time.time() - iter_report_start
				eta = str(datetime.timedelta(seconds=int(iter_report_time * (expect_iter - iter_cnt) / iter_cnt))).split(".")[0]
				self.logger.info(f"ETA:{eta} " + log_info)
				
				wandb_log_dict = {
					'epoch': epoch,
					'val/student_mean_acc': student_mean_acc,
					'val/student_mean_best_acc': student_max_mean_acc
				}
				for target_idx in range(n_target):
					domain_name = target_domain_names[target_idx]
					wandb_log_dict[f"val/student_{domain_name}_acc"] = student_target_accs[target_idx]
					wandb_log_dict[f"val/student_{domain_name}_best_acc"] = student_max_accs[target_idx]
				
				wandb.log(wandb_log_dict)
		
		self.logger.info('######### End! ##########')





