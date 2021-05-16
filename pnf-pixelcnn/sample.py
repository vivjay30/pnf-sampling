import time
import os
import argparse
import cv2
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
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='data', help='Location for the dataset')
parser.add_argument('-d', '--dataset', type=str,
                    default='cifar', help='Can be either cifar|mnist')
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
parser.add_argument('-c', '--count', type=int, default=1,
                    help='Number of separations')
parser.add_argument('-T', '--iterations', type=int, default=300,
                    help='Number of Langevin iterations')
args = parser.parse_args()

# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

checkpoints = { 1.0        : ('pcnn_input_sigma:1.00000_lr:0.00040_nr-resnet5_nr-filters160_9.pth', 1),
                0.77426368 : ('pcnn_input_sigma:0.77426_lr:0.00040_nr-resnet5_nr-filters160_9.pth', 1),
                0.59948425 : ('pcnn_input_sigma:0.59948_lr:0.00040_nr-resnet5_nr-filters160_9.pth', 1),
                0.46415888 : ('pcnn_input_sigma:0.46416_lr:0.00040_nr-resnet5_nr-filters160_9.pth', 1),
                0.35938137 : ('pcnn_input_sigma:0.35938_lr:0.00040_nr-resnet5_nr-filters160_9.pth', 1),
                0.27825594 : ('pcnn_input_sigma:0.27826_lr:0.00040_nr-resnet5_nr-filters160_9.pth', 1),
                0.21544347 : ('pcnn_input_sigma:0.21544_lr:0.00040_nr-resnet5_nr-filters160_9.pth', 1),
                0.16681005 : ('pcnn_input_sigma:0.16681_lr:0.00040_nr-resnet5_nr-filters160_9.pth', 1),
                0.12915497 : ('pcnn_input_sigma:0.12915_lr:0.00040_nr-resnet5_nr-filters160_9.pth', 1),
                0.1        : ('pcnn_input_sigma:0.10000_lr:0.00040_nr-resnet5_nr-filters160_9.pth', 1),
                0.07742637 : ('pcnn_input_sigma:0.07743_lr:0.00040_nr-resnet5_nr-filters160_9.pth', 1),
                0.05994843 : ('pcnn_input_sigma:0.05995_lr:0.00040_nr-resnet5_nr-filters160_9.pth', 1),
                0.04641589 : ('pcnn_input_sigma:0.04642_lr:0.00040_nr-resnet5_nr-filters160_9.pth', 1),
                0.03593814 : ('pcnn_input_sigma:0.03594_lr:0.00040_nr-resnet5_nr-filters160_9.pth', 1),
                0.02782559 : ('pcnn_input_sigma:0.02783_lr:0.00040_nr-resnet5_nr-filters160_9.pth', 1),
                0.02154435 : ('pcnn_input_sigma:0.02154_lr:0.00040_nr-resnet5_nr-filters160_9.pth', 1),
                0.01668101 : ('pcnn_input_sigma:0.01668_lr:0.00040_nr-resnet5_nr-filters160_9.pth', 1),
                0.01291550 : ('pcnn_input_sigma:0.01292_lr:0.00040_nr-resnet5_nr-filters160_9.pth', 1),
                0.01       : ('pcnn_input_sigma:0.01000_lr:0.00040_nr-resnet5_nr-filters160_9.pth', 1),
              }


sample_batch_size = 25
obs = (1, 28, 28) if 'mnist' in args.dataset else (3, 32, 32)
input_channels = obs[0]
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

if 'mnist' in args.dataset : 
    test_loader  = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)

elif 'cifar' in args.dataset : 
    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)
else :
    raise Exception('{} dataset not in {mnist, cifar10}'.format(args.dataset))

def write_images(x, fname, itm, sigma, n=1):
    x = x.permute(0,2,3,1).numpy()
    d = x.shape[1]
    panel = np.zeros([n*d,n*d,3],dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            panel[i*d:(i+1)*d,j*d:(j+1)*d,:] = (255*(x[i*n+j]/2.+.5)).clip(0,255).astype(np.uint8)[:,:,::-1]

    cv2.imwrite(os.path.join('images','pcnn_{}{}_{}_{}.png'.format(fname,args.seed,itm,sigma)), panel)

model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
model = model.cuda()
model.eval()

test_loader = iter(test_loader)
nll_history = []
smoothed_history = []
for itm in range(args.count):
    x0 = torch.randn(1, 3, 32, 32).cuda()

    n_steps_each = args.iterations
    step_lr=args.lr
    for sigma, (checkpoint, step_multiplier) in checkpoints.items():
        load_part_of_model(model, os.path.join(args.load_params, checkpoint), prefix='')
    
        eta = step_lr * (sigma / .01)**2
        print('sigma = {}'.format(sigma))
        for i, step in enumerate(range(n_steps_each*step_multiplier)):
            noise_x0 = torch.randn_like(x0) * np.sqrt(eta*2)

            x0 = x0.detach()
            x0.requires_grad=True
            nll0 = discretized_mix_logistic_loss_smoothed_nll(x0, model(x0), sigma, h=256)
            grad_x0 = torch.autograd.grad(-nll0.sum(), x0)[0]

            x0 = x0 + eta * grad_x0 + noise_x0

            norm0 = np.linalg.norm(grad_x0.contiguous().view(-1,3*32*32).cpu().numpy(),axis=1).mean()

            smoothed_history.append(nll0.mean().detach().cpu().numpy()/np.log(2.))

        pretrained = 'pcnn_lr.0.00040_nr-resnet5_nr-filters160_889.pth'
        load_part_of_model(model, os.path.join('pretrained', pretrained))

        with torch.no_grad(): xhat0 = model(x0)
        unsmoothed_nll0 = discretized_mix_logistic_loss(x0, xhat0)/(np.prod(x0.shape) * np.log(2.))
        nll_history.append(unsmoothed_nll0.detach().cpu().numpy())

        fmt = 'step: {}, nll0: {}, smooth_nll0: {}, |norm0|: {}'
        print(fmt.format(i, unsmoothed_nll0, nll0.mean()/np.log(2.), norm0))

        write_images(x0.detach().cpu(), 'x', itm, sigma)

np.save('opt/unsmoothed_nll_more_lr{}_T{}.npy'.format(step_lr, n_steps_each), nll_history)
np.save('opt/nll_more_lr{}_T{}.npy'.format(step_lr, n_steps_each), smoothed_history)

