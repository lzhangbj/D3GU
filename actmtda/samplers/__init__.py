from .target_combined_samplers import *

from actmtda.utils.registry import SAMPLER


def build_sampler(cfg, source_dataset, target_dataset, save_root, logger):
	assert cfg.SAMPLER.TYPE in SAMPLER, \
		"cfg.SAMPLER.TYPE: {} are not registered in registry".format(
			cfg.SAMPLER.TYPE
		)
	sampler = SAMPLER[cfg.SAMPLER.TYPE](cfg, source_dataset, target_dataset, save_root, logger=logger)
	
	return sampler
