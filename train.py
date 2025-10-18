from utils.regression_trainer import RegTrainer
import argparse
import os
import torch
import numpy as np
import random
args = None


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.use_deterministic_algorithms(True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--seed', type=int, default=3407,
                        help='random seed')
    parser.add_argument('--data-dir', default=r"",
                        help='training data directory')
    parser.add_argument('--save-dir', default=r"",
                        help='directory to save models.')
    parser.add_argument('--pretrained-model', default=r"/pretrained/vgg_vit_depth_2_head_6.pth",
                        help='directory of pre-trained weight.')

    parser.add_argument('--lr', type=float, default=1e-5,
                        help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='the weight decay')
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=400,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=0,
                        help='the epoch start to val')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')

    parser.add_argument('--crop-size', type=int, default=224,
                        help='the crop size of the train image')
    parser.add_argument('--downsample-ratio', type=int, default=8,
                        help='downsample ratio')

    parser.add_argument('--use-background', type=bool, default=True,
                        help='whether to use background modelling')
    parser.add_argument('--sigma', type=float, default=8.0,
                        help='sigma for likelihood')
    parser.add_argument('--background-ratio', type=float, default=0.15,
                        help='background ratio')


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()
