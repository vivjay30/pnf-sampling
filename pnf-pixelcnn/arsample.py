import time
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
from tensorboardX import SummaryWriter
from utils import * 
from model import * 
from PIL import Image

parser = argparse.ArgumentParser()
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-l', '--lr', type=float,
                    default=0.0002, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=16,
                    help='Batch size during training per GPU')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')
parser.add_argument('-g', '--sigma', type=float,
                    default=0.01, help='Standard deviation of additive Gaussian noise.')
args = parser.parse_args()

# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

sample_batch_size = 25
obs = (3, 32, 32)
input_channels = obs[0]
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
model = model.cuda()

if args.sigma == 0.0:
    pretrained = 'pcnn_lr.0.00040_nr-resnet5_nr-filters160_889.pth'
    print('Loading pretrained {}'.format(pretrained))
    load_part_of_model(model, os.path.join('pretrained', pretrained))
else:
    checkpoint = 'pcnn_input_sigma:{:.5f}_lr:0.00040_nr-resnet5_nr-filters160_9.pth'.format(args.sigma)
    print('Loading checkpoint {}'.format(checkpoint))
    load_part_of_model(model, os.path.join('finetuned', checkpoint), prefix='')

def sample(model):
    model.train(False)
    data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
    data = data.cuda()
    for i in range(obs[1]):
        for j in range(obs[2]):
            data_v = Variable(data, volatile=True)
            out   = model(data_v, sample=True)
            out_sample = sample_from_discretized_mix_logistic(out, args.nr_logistic_mix)
            data[:, :, i, j] = out_sample.data[:, :, i, j] + args.sigma*torch.randn(out_sample.data[:, :, i, j].shape).cuda()
    return data

with torch.no_grad():
    for i in range(10):
        x0 = sample(model)
        #x0 = rescaling_inv(x0)
        unsmoothed_nll0 = discretized_mix_logistic_loss(x0, model(x0))/(np.prod(x0.shape) * np.log(2.))
        print(unsmoothed_nll0)
