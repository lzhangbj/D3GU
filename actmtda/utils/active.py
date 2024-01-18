import os
import numpy as np

def save_labeled_set(labeled_set, dataset, save_path="*.npy"):
	idx_pairs = decode_labeled_set(labeled_set, dataset)
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	np.save(save_path, idx_pairs)
	
def decode_labeled_set(labeled_set, dataset):
	idx_pairs = [list(dataset.get_original_idx(idx)) for idx in labeled_set]
	idx_pairs = np.array(idx_pairs, dtype=np.int32)  # (N, 2) dataset idx + sample idx
	return idx_pairs

def read_labeled_set(dataset, read_path="*.npy"):
	idx_pairs = np.load(read_path)
	idx = np.array([dataset.get_concat_idx(idx_pair[0], idx_pair[1]) for idx_pair in idx_pairs])
	return idx
	

