import time
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from utils import * 
from model import * 
from PIL import Image

parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='models',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--dataset', type=str,
                    default='cifar', help='Can be either cifar|mnist')
parser.add_argument('-p', '--print_every', type=int, default=50,
                    help='how many iterations between print statements')
parser.add_argument('-t', '--save_interval', type=int, default=10,
                    help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', type=str, default=None,
                    help='Restore training from previous model checkpoint?')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-l', '--lr', type=float,
                    default=0.0002, help='Base learning rate')
parser.add_argument('-b', '--batch_size', type=int, default=16,
                    help='Batch size during training per GPU')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')
parser.add_argument('-g', '--sigma', type=float,
                    default=0.0, help='Standard deviation of additive Gaussian noise.')
args = parser.parse_args()

# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

model_name = 'pcnn_input_sigma:{:.5f}_lr:{:.5f}_nr-resnet{}_nr-filters{}'.format(args.sigma,args.lr, args.nr_resnet, args.nr_filters)
assert not os.path.exists(os.path.join('runs', model_name)), '{} already exists!'.format(model_name)

sample_batch_size = 25
obs = (3, 32, 32)
input_channels = obs[0]
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
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

model.eval()
test_loss = 0.
rounded_loss = 0.
with torch.no_grad():
    for batch_idx, (input,_) in enumerate(test_loader):
        input = input.cuda(non_blocking=True) + args.sigma*torch.randn(*input.shape).cuda()
        test_loss += discretized_mix_logistic_loss(input, model(input)).item()

        #rounded_input = rescaling_inv(torch.round(255*rescaling(input))/255.)
        #rounded_loss += discretized_mix_logistic_loss(rounded_input, model(rounded_input)).item()

deno = batch_idx * args.batch_size * np.prod(obs) * np.log(2.)
print('test loss : %s' % (test_loss / deno))
#print('rounded test loss : %s' % (rounded_loss / deno))

