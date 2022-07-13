import argparse
import os
"""
args
"""

parser = argparse.ArgumentParser(description='RotationNet')

# Logging
parser.add_argument(
    '--results_dir',
    metavar='RESULTS_DIR',
    default='../models/ImageNet',
    help='results dir')

parser.add_argument(
    '--save',
    metavar='SAVE',
    default='20220712v1',
    help='saved folder (named by datetime)')

parser.add_argument(
    '--resume',
    dest='resume',
    action='store_true',
    help='resume to latest checkpoint')

parser.add_argument(
    '-e',
    '--evaluate',
    type=str,
    metavar='FILE',
    help='evaluate model FILE on validation set')

parser.add_argument(
    '--seed', 
    default=None,
    type=int, 
    help='random seed')

parser.add_argument(
    '--model',
    '-a',
    metavar='MODEL',
    default='resnet18_1w1a_IR',
    help='model architecture ')

parser.add_argument(
    '--model_fp',
    default='resnet18_fp',
    help='teacher model architecture ')

parser.add_argument(
    '--dataset',
    default='imagenet',
    type=str,
    help='dataset, default:imagenet')

parser.add_argument(
    '--data_path',
    type=str,
    default='/data/shangyuzhang/imagenet/images',
    help='The dictionary where the dataset is stored.')

parser.add_argument(
    '--type',
    default='torch.cuda.FloatTensor',
    help='type of tensor - e.g torch.cuda.FloatTensor')

# Training
parser.add_argument(
    '--gpus',
    default='0',
    help='gpus used for training - e.g 0,1,3')

parser.add_argument(
    '--lr', 
    default=0.1, 
    type=float, 
    help='learning rate')

parser.add_argument(
    '--weight_decay',
    type=float,
    default=1e-4,
    help='Weight decay of loss. default:1e-4')

parser.add_argument(
    '--momentum',
    default=0.9, 
    type=float, 
    metavar='M',
    help='momentum')

parser.add_argument(
    '--workers',
    default=10,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 8)')

parser.add_argument(
    '--alpha',
    default=8,
    type=float,
    metavar='N',
    help='alpha to adjust the weight of LCR loss in the overall loss function')

parser.add_argument(
    '--epochs',
    default=100,
    type=int,
    metavar='N',
    help='number of total epochs to run')

parser.add_argument(
    '--start_epoch',
    default=-1,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')

parser.add_argument(
    '-b',
    '--batch_size',
    default=200,
    type=int,
    metavar='N',
    help='mini-batch size for training (default: 200)')

parser.add_argument(
    '-bt',
    '--batch_size_test',
    default=128,
    type=int,
    help='mini-batch size for testing (default: 128)')

parser.add_argument(
    '--print_freq',
    '-p',
    default=1000,
    type=int,
    metavar='N',
    help='print frequency (default: 500)')

parser.add_argument(
    '--time_estimate',
    default=1,
    type=int,
    metavar='N',
    help='print estimating finish time,set to 0 to disable')

parser.add_argument(
    '--rotation_update',
    default=1,
    type=int,
    metavar='N',
    help='interval of updating rotation matrix (default:1)')

parser.add_argument(
    '--Tmin',
    default=1e-3,
    type=float, 
    metavar='M',
    help='minimum of T (default:1e-2)')

parser.add_argument(
    '--Tmax',
    default=1e1, 
    type=float, 
    metavar='M',
    help='maximum of T (default:1e1)')

parser.add_argument(
    '--lr_type',
    type=str,
    default='cos',
    help='choose lr_scheduler,(default:cos)')

parser.add_argument(
    '--lr_decay_step',
    nargs='+',
    type=int,
    help='lr decay step for MultiStepLR')

parser.add_argument(
    '--a32',
    dest='a32',
    action='store_true',
    help='w1a32')

parser.add_argument(
    '--warm_up',
    dest='warm_up',
    action='store_true',
    help='use warm up or not')

parser.add_argument(
    '--use_dali',
    dest='use_dali',
    action='store_true',
    help='use DALI to load dataset or not') 

args = parser.parse_args()