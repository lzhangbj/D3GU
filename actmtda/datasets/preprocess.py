import torch
from torchvision import transforms

def image_train(resize_size=256, crop_size=224):
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	return transforms.Compose([
		ResizeImage(resize_size),
		transforms.RandomResizedCrop(crop_size),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize,
	])


def image_test(resize_size=256, crop_size=224):
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	return transforms.Compose([
		ResizeImage(resize_size),
		transforms.CenterCrop(crop_size),
		transforms.ToTensor(),
		normalize
	])


class ResizeImage:
	def __init__(self, size):
		if isinstance(size, int):
			self.size = (int(size), int(size))
		else:
			self.size = size
	
	def __call__(self, img):
		th, tw = self.size
		return img.resize((th, tw))


class PlaceCrop:
	"""Crops the given PIL.Image at the particular index.
	Args:
		size (sequence or int): Desired output size of the crop. If size is an
			int instead of sequence like (w, h), a square crop (size, size) is
			made.
	"""
	
	def __init__(self, size, start_x, start_y):
		if isinstance(size, int):
			self.size = (int(size), int(size))
		else:
			self.size = size
		self.start_x = start_x
		self.start_y = start_y
	
	def __call__(self, img):
		"""
		Args:
			img (PIL.Image): Image to be cropped.
		Returns:
			PIL.Image: Cropped image.
		"""
		th, tw = self.size
		return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))

class SubsetSequentialSampler(torch.utils.data.Sampler):
	r"""Samples elements sequentially from a given list of indices, without replacement.
	Arguments:
		indices (sequence): a sequence of indices
	"""

	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return (self.indices[i] for i in range(len(self.indices)))

	def __len__(self):
		return len(self.indices)