import os
from PIL import Image
import warnings

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import numpy as np
import json
import bisect
from .preprocess import image_train, image_test, SubsetSequentialSampler


"""
Code adapted from CDAN github repository.
https://github.com/thuml/CDAN/tree/master/pytorch
"""

class ConcatDataset(Dataset):
	"""
	Dataset to concatenate multiple datasets.
	Purpose: useful to assemble different existing datasets, possibly
	large-scale datasets as the concatenation operation is done in an
	on-the-fly manner.

	Arguments:
		datasets (sequence): List of datasets to be concatenated
	"""
	
	@staticmethod
	def cumsum(sequence):
		r, s = [], 0
		for e in sequence:
			l = len(e)
			r.append(l + s)
			s += l
		return r
	
	def __init__(self, datasets):
		super(ConcatDataset, self).__init__()
		assert len(datasets) > 0, 'datasets should not be an empty iterable'
		self.datasets = list(datasets)
		self.domain_names = [dt.domain_name for dt in datasets]
		self.cumulative_sizes = self.cumsum(self.datasets)
		self.imgs = []
		for dataset in datasets:
			self.imgs += dataset.imgs
	
	def __len__(self):
		return self.cumulative_sizes[-1]
	
	def __getitem__(self, idx):
		dataset_idx, sample_idx = self.get_original_idx_from_concat_idx(idx)
		batch = self.datasets[dataset_idx][sample_idx]
		return batch
	
	@property
	def cummulative_sizes(self):
		warnings.warn('cumulative_sizes attribute is renamed to '
		              'cumulative_sizes', DeprecationWarning, stacklevel=2)
		return self.cumulative_sizes

	def get_original_idx_from_concat_idx(self, idx):
		dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
		prev_cum_size = 0 if dataset_idx==0 else self.cumulative_sizes[dataset_idx-1]
		return dataset_idx, idx - prev_cum_size
	
	def get_concat_idx_from_original_idx(self, dataset_idx, sample_idx):
		start = 0
		if dataset_idx > 0:
			start = self.cumulative_sizes[dataset_idx-1]
		return start + sample_idx
	
	def get_image_idx(self, image_ids):
		x = np.array([tup[0] for tup in self.imgs], dtype=np.int32)
		y = np.array(image_ids).astype(np.int32)
		
		xsorted = np.argsort(x)
		ypos = np.searchsorted(x[xsorted], y)
		indices = xsorted[ypos]
		return indices.tolist()
	
	
class ImageList(Dataset):
	def __init__(self, image_root, image_list_root,
				dataset_name, dataset_mapping_file,
				domain_name, domain_label, split='train',
	            labeled_set=None,
				transform=None):
		'''
		Args:
			image_root:
			image_list_root: directory of dataset where images are kept
			dataset_name:
			dataset_mapping_file:
			domain_name:
			domain_label:
			split:
			labeled_set: list of image ids
			transform:
		'''
		self.image_root = image_root
		self.image_list_root = image_list_root
		self.dataset_name = dataset_name  # name of whole dataset
		self.domain_name = domain_name  # name of the domain
		self.domain_label = domain_label
		self.transform = transform
		self.loader = self._rgb_loader
		with open(dataset_mapping_file, 'r') as f:
			self.dataset_mapping = json.load(f)
		if split is not None:
			self.imgs = self._make_dataset(os.path.join(image_list_root, domain_name + '_' + split + '.txt'), labeled_set)
		else:
			self.imgs = self._make_dataset(os.path.join(image_list_root, domain_name + '.txt'), labeled_set)
		
	def _rgb_loader(self, path):
		with open(path, 'rb') as f:
			with Image.open(f) as img:
				return img.convert('RGB')
	
	def _make_dataset(self, image_list_path, labeled_set=None):
		image_list = open(image_list_path).readlines()
		# list of (image_id, image_name, class) tuple
		images = []
		for line in image_list:
			image_name = line.strip().split()[0]
			image_class = int(line.strip().split()[1])
			image_id = self.dataset_mapping[image_name]
			if labeled_set is not None and image_id not in labeled_set:
				continue
			images.append((image_id, image_name, image_class))
		return images
	
	def __getitem__(self, index):
		output = {}
		image_id, file_name, target = self.imgs[index]
		img = self.loader(os.path.join(self.image_root, file_name))
		if self.transform is not None:
			img = self.transform(img)
		
		output['image'] = img
		output['image_id'] = image_id
		output['file_name'] = file_name
		output['target'] = target
		output['domain'] = self.domain_label
		
		return output
	
	def __len__(self):
		return len(self.imgs)


class SemiImageList(Dataset):
	def __init__(self, image_root, image_list_root,
	             num_class,
	             dataset_name, dataset_mapping_file,
	             domain_name, domain_label, split='train',
	             labeled_set=None,
	             pseudolabeled_set=[],
	             pseudolabeled_labels=[],
	             pseudolabeled_weights=[],
	             transform=None):
		'''
		Args:
			image_root:
			image_list_root: directory of dataset where images are kept
			dataset_name:
			dataset_mapping_file:
			domain_name:
			domain_label:
			split:
			labeled_set: list of image ids
			transform:
		'''
		self.image_root = image_root
		self.image_list_root = image_list_root
		self.dataset_name = dataset_name  # name of whole dataset
		self.num_class = num_class
		self.domain_name = domain_name  # name of the domain
		self.domain_label = domain_label
		self.transform = transform
		self.loader = self._rgb_loader
		with open(dataset_mapping_file, 'r') as f:
			self.dataset_mapping = json.load(f)
		if split is not None:
			self.imgs = self._make_dataset(os.path.join(image_list_root, domain_name + '_' + split + '.txt'),
			                               labeled_set, pseudolabeled_set, pseudolabeled_labels, pseudolabeled_weights)
		else:
			self.imgs = self._make_dataset(os.path.join(image_list_root, domain_name + '.txt'),
			                               labeled_set, pseudolabeled_set, pseudolabeled_labels, pseudolabeled_weights)
	
	def _rgb_loader(self, path):
		with open(path, 'rb') as f:
			with Image.open(f) as img:
				return img.convert('RGB')
	
	def _make_dataset(self, image_list_path, 
	                  labeled_set=None,
	                  pseudolabeled_set=[],
	                  pseudolabeled_labels=[],
	                  pseudolabeled_weights=[]):
		image_list = open(image_list_path).readlines()
		# list of (image_id, image_name, class) tuple
		images = []
		for line in image_list:
			image_name = line.strip().split()[0]
			image_class = int(line.strip().split()[1])
			image_id = self.dataset_mapping[image_name]
			if labeled_set is None or image_id in labeled_set:
				label = np.zeros(self.num_class, dtype=np.float32)
				label[image_class] = 1
				images.append((image_id, image_name, label, 1.0))
			elif image_id in pseudolabeled_set:
				image_pseudo_idx = pseudolabeled_set.index(image_id)
				pseudo_label = pseudolabeled_labels[image_pseudo_idx]
				pseudo_weight = pseudolabeled_weights[image_pseudo_idx]
				images.append((image_id, image_name, pseudo_label, pseudo_weight))
			
		return images
	
	def __getitem__(self, index):
		output = {}
		image_id, file_name, target, weight = self.imgs[index]
		img = self.loader(os.path.join(self.image_root, file_name))
		if self.transform is not None:
			img = self.transform(img)
		
		output['image'] = img
		output['image_id'] = image_id
		output['file_name'] = file_name
		output['target'] = torch.from_numpy(target).float()
		output['weight'] = weight
		output['domain'] = self.domain_label
		
		return output
	
	def __len__(self):
		return len(self.imgs)


def build_dataset(dataset_cfg, domain_names, domain_label_start=0, split='train', labeled_set=None, is_train=False):
	
	dataset_list = []
	transform = image_train() if is_train else image_test()
	
	for i, domain_name in enumerate(domain_names):
		domain_label = i+domain_label_start
		dataset = ImageList(image_root=dataset_cfg.DATAROOT,
							image_list_root=dataset_cfg.IMAGELIST_ROOT,
							dataset_name=dataset_cfg.DATASET_NAME,
							dataset_mapping_file=dataset_cfg.DATASET_MAPPING_PATH,
							domain_name=domain_name,
							domain_label=domain_label,
							labeled_set=labeled_set,
							split=split,
							transform=transform)
		dataset_list.append(dataset)
	concat_dataset = ConcatDataset(dataset_list)
	
	return concat_dataset


def build_semi_dataset(dataset_cfg, domain_names, domain_label_start=0, split='train',
                       labeled_set=None,
                       pseudolabeled_set=[],
                       pseudolabeled_labels=[],
                       pseudolabeled_weights=[],
                       is_train=False):
	dataset_list = []
	transform = image_train() if is_train else image_test()
	
	for i, domain_name in enumerate(domain_names):
		domain_label = i + domain_label_start
		dataset = SemiImageList(image_root=dataset_cfg.DATAROOT,
		                    num_class=dataset_cfg.NUM_CLASS,
		                    image_list_root=dataset_cfg.IMAGELIST_ROOT,
		                    dataset_name=dataset_cfg.DATASET_NAME,
		                    dataset_mapping_file=dataset_cfg.DATASET_MAPPING_PATH,
		                    domain_name=domain_name,
		                    domain_label=domain_label,
		                    labeled_set=labeled_set,
		                    pseudolabeled_set=pseudolabeled_set,
	                        pseudolabeled_labels=pseudolabeled_labels,
	                        pseudolabeled_weights=pseudolabeled_weights,
		                    split=split,
		                    transform=transform)
		dataset_list.append(dataset)
	concat_dataset = ConcatDataset(dataset_list)
	
	return concat_dataset


def build_dataloader(dataset, batchsize, num_worker, is_train=False, labeled_set=None, drop_last=None, sample_weights=None):
	'''
	labeled_set: indices of labeled data in dataset
	'''
	if sample_weights is not None:
		assert labeled_set is None, "weighted sampler is only used for source as in lambda"
		assert is_train, "weighted sampler is only used for source training as in lambda"
		sampler = WeightedRandomSampler(sample_weights, len(dataset), replacement=True)
		dataloader = DataLoader(dataset,
		                        batch_size=batchsize,
		                        sampler=sampler,
		                        num_workers=num_worker,
		                        drop_last=True,
		                        pin_memory=True)
		return dataloader

	sampler = SubsetRandomSampler if is_train else SubsetSequentialSampler
	if labeled_set is None:
		dataloader = DataLoader(dataset,
								batch_size=batchsize,
								sampler=sampler(range(len(dataset))),
								num_workers=num_worker,
								drop_last=drop_last if drop_last is not None else is_train,
								pin_memory=True)
	else:
		dataloader = DataLoader(dataset,
		                        batch_size=batchsize,
		                        sampler=sampler(labeled_set),
		                        num_workers=num_worker,
		                        drop_last=drop_last if drop_last is not None else is_train,
		                        pin_memory=True)
	return dataloader