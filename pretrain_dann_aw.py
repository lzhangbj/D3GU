import torch
import argparse
import os
import torch.nn as nn
import pdb

import datetime
import wandb

from actmtda.utils.default import cfg
from actmtda.apis.pretrain.pretrain_dann_aw_trainer import PretrainDANNAwTrainer

parser = argparse.ArgumentParser(description="PyTorch sseg")
parser.add_argument(
	"--config_file",
	metavar="FILE",
	help="path to config file",
)
parser.add_argument("--resume_from", type=str, default=None)
parser.add_argument("--work_dir", type=str, default=None)
parser.add_argument("--seed", type=int, default=-1)

args = parser.parse_args()

if __name__ == "__main__":

	cfg.merge_from_file(args.config_file)
	if args.seed != -1:
		cfg.RANDOM_SEED = args.seed
		
	if args.resume_from:
		cfg.TRAIN.RESUME_FROM = args.resume_from
	if args.work_dir:
		cfg.WORK_DIR = args.work_dir
	elif cfg.WORK_DIR == "":
		config_name = "/".join(args.config_file.split('/')[1:]).replace(".yaml", "")
		cfg.WORK_DIR = os.path.join("exps", config_name)
	cfg.freeze()
	
	dir_cp = cfg.WORK_DIR
	if not os.path.exists(dir_cp):
		os.makedirs(dir_cp)
	os.makedirs(os.path.join(cfg.WORK_DIR, 'ckpt'), exist_ok=True)
	
	wandb_project = "_".join(args.config_file.split('/')[1:3])
	wandb_name = "/".join(args.config_file.split('/')[3:]).replace('.yaml',"") \
	             + "-" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
	
	wandb.init(project=wandb_project, name=wandb_name)
	
	trainer = PretrainDANNAwTrainer(cfg)
	trainer.run()




