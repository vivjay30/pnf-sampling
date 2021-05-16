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
                    default=3e-6, help='Base learning rate')
parser.add_argument('-b', '--batch_size', type=int, default=16,
                    help='Batch size during training per GPU')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')
parser.add_argument('-c', '--count', type=int, default=1,
                    help='Number of separations')
parser.add_argument('-T', '--iterations', type=int, default=300,
                    help='Number of Langevin iterations per noise level')
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
                0.01       : ('pcnn_input_sigma:0.01000_lr:0.00040_nr-resnet5_nr-filters160_9.pth', 1) }


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

def sample(model):
    model.train(False)
    data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
    data = data.cuda()
    for i in range(obs[1]):
        for j in range(obs[2]):
            data_v = Variable(data, volatile=True)
            out   = model(data_v, sample=True)
            out_sample = sample_op(out)
            data[:, :, i, j] = out_sample.data[:, :, i, j]
    return data

def evaluate(model, sigma=0):
    torch.cuda.synchronize()
    model.eval()
    test_loss = 0.
    for batch_idx, (input,_) in enumerate(test_loader):
        input = input.cuda(non_blocking=True)
        input_var = Variable(input)
        output = model(input_var)
        if sigma == 0: loss = discretized_mix_logistic_loss(x0, model(x0))
        else: loss = discretized_mix_logistic_loss_smoothed(input_var, output, sigma).sum()
        test_loss += loss.item()
        del loss, output

    deno = batch_idx * args.batch_size * np.prod(obs) * np.log(2.)
    return test_loss / deno

def write_images(x, fname, itm, n=2):
    x = x.permute(0,2,3,1).numpy()
    d = x.shape[1]
    panel = np.zeros([n*d,n*d,3],dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            panel[i*d:(i+1)*d,j*d:(j+1)*d,:] = (255*(x[i*n+j]/2.+.5)).clip(0,255).astype(np.uint8)[:,:,::-1]

    cv2.imwrite(os.path.join('images','pcnn_{}{}_{}.png'.format(fname,args.seed,itm)), panel)

# 0 - plane, 1 - car, 8 - ships, 9 - trucks
# 2 - bird, 3 - cat, 4 - deer, 5 - dog, 6 - frog, 7 - horse
def generate_mixture_split(loader, labels0=[0,1,8,9],labels1=[2,3,4,5,6,7]):
    gt0 = []
    gt1 = []

    while (len(gt0) < args.batch_size) or (len(gt1) < args.batch_size):
        x,y = next(loader)
        for idx in range(args.batch_size):
            if y[idx] in labels0 and (len(gt0) < args.batch_size): gt0.append(x[idx])
            elif y[idx] in labels1 and (len(gt1) < args.batch_size): gt1.append(x[idx])

    gt0 = torch.stack(gt0, axis=0)
    gt1 = torch.stack(gt1, axis=0)

    mixed = gt0 + gt1

    x0 = max(checkpoints.keys())*torch.randn(*gt0.shape)
    x1 = max(checkpoints.keys())*torch.randn(*gt1.shape)

    return mixed.cuda(), x0.cuda(), x1.cuda(), gt0.cuda(), gt1.cuda()

def generate_mixture(loader):
    gt0,y0 = next(loader)
    gt1,y1 = next(loader)
    mixed = gt0 + gt1

    x0 = max(checkpoints.keys())*torch.randn(*gt0.shape)
    x1 = max(checkpoints.keys())*torch.randn(*gt1.shape)

    return mixed.cuda(), x0.cuda(), x1.cuda(), gt0.cuda(), gt1.cuda()

model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
model = model.cuda()
model.eval()

nll_history = []
smoothed_history = []
for itm in range(args.count):
    mixed, x0, x1, gt0, gt1 = generate_mixture_split(iter(test_loader))
    write_images(mixed.cpu()/2., 'mixed', itm)
    write_images(gt0.cpu(), 'xgt', itm)
    write_images(gt1.cpu(), 'ygt', itm)

    n_steps_each = args.iterations#300
    step_lr = args.lr
    for sigma, (checkpoint, step_multiplier) in sorted(checkpoints.items(), reverse=True):
        load_part_of_model(model, os.path.join(args.load_params, checkpoint), prefix='')
    
        eta = step_lr * (sigma / .01)**2
        gamma = 1.0 / (sigma**2)
        print('sigma = {}'.format(sigma))
        for i, step in enumerate(range(n_steps_each*step_multiplier)):
            noise_x0 = torch.randn_like(x0) * np.sqrt(eta*2)
            noise_x1 = torch.randn_like(x1) * np.sqrt(eta*2)

            x0 = x0.detach()
            x0.requires_grad=True
            nll0 = discretized_mix_logistic_loss_smoothed_nll(x0, model(x0), sigma, h=64)
            grad_x0 = torch.autograd.grad(-nll0.sum(), x0)[0]

            x1 = x1.detach()
            x1.requires_grad=True
            nll1 = discretized_mix_logistic_loss_smoothed_nll(x1, model(x1), sigma, h=64)
            grad_x1 = torch.autograd.grad(-nll1.sum(), x1)[0]

            x0 = x0 + eta * (grad_x0 - gamma * (x0 + x1 - mixed)) + noise_x0
            x1 = x1 + eta * (grad_x1 - gamma * (x0 + x1 - mixed)) + noise_x1

            norm0 = np.linalg.norm(grad_x0.contiguous().view(-1,3*32*32).cpu().numpy(),axis=1).mean()
            norm1 = np.linalg.norm(grad_x1.contiguous().view(-1,3*32*32).cpu().numpy(),axis=1).mean()
            recon = ((x0 + x1 - mixed)**2).contiguous().view(-1,3*32*32).sum(1).mean()

            nll = (nll0 + nll1) / 2.
            smoothed_history.append(nll.mean().detach().cpu().numpy()/np.log(2.))

        pretrained = 'pcnn_lr.0.00040_nr-resnet5_nr-filters160_889.pth'
        load_part_of_model(model, os.path.join('pretrained', pretrained))

        with torch.no_grad(): xhat0 = model(x0)
        unsmoothed_nll0 = discretized_mix_logistic_loss(x0, xhat0)/(np.prod(x0.shape) * np.log(2.))

        with torch.no_grad(): xhat1 = model(x1)
        unsmoothed_nll1 = discretized_mix_logistic_loss(x1, xhat1)/(np.prod(x1.shape) * np.log(2.))

        fmt = 'step: {}, recon: {}, nll0: {}, smooth_nll0: {}, |norm1|: {}, nll1: {}, smooth_nll1: {} |norm2|: {}'
        print(fmt.format(i,recon, unsmoothed_nll0, nll0.mean()/np.log(2),norm0, unsmoothed_nll1, nll1.mean()/np.log(2),norm1))

        write_images(x0.detach().cpu(), 'x', itm)
        write_images(x1.detach().cpu(), 'y', itm)
        write_images((x0/2.+x1/2.).detach().cpu(), 'recon', itm)

        nll = (unsmoothed_nll0 + unsmoothed_nll1) / 2.
        nll_history.append(nll.detach().cpu().numpy())

    #print('test loss : %s' % evaluate(model, sigma))

np.save('opt/sep_unsmoothed_nll_lr{}_T{}.npy'.format(step_lr, n_steps_each), nll_history)
np.save('opt/sep_nll_lr{}_T{}.npy'.format(step_lr, n_steps_each), smoothed_history)
