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
parser.add_argument('-s', '--seed', type=int, default=0,
                    help='Random seed to use')
parser.add_argument('-c', '--count', type=int, default=1,
                    help='Number of separations')
parser.add_argument('-u', '--downsample', type=int, default=2,
                    help='Downsampling factor')
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

def write_images(x, fname, itm, n=4):
    x = x.permute(0,2,3,1).numpy()
    d = x.shape[1]
    panel = np.zeros([n*d,n*d,3],dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            panel[i*d:(i+1)*d,j*d:(j+1)*d,:] = (255*(x[i*n+j]/2.+.5)).clip(0,255).astype(np.uint8)[:,:,::-1]

    cv2.imwrite(os.path.join('images','pcnn_super_{}{}_{}.png'.format(fname,args.seed,itm)), panel)

def generate_downres(loader, d=2):
    gt,y = next(loader)

    x = torch.randn(*gt.shape)
    x[:,:,::d,::d] = gt[:,:,::d,::d].clone()
    #x = gt.clone()
    #x[:,:,::d,::d] = 0
    
    down = -1*torch.ones(*gt.shape)
    down[:,:,::d,::d] = gt[:,:,::d,::d].clone()

    return down.cuda(), x.cuda(), gt.cuda()

model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
model = model.cuda()
model.eval()

nll_history = []
smoothed_history = []
d = args.downsample
for itm in range(args.count):
    down, x, gt = generate_downres(iter(test_loader), d=d)
    write_images(down.cpu(), 'down', itm)
    write_images(gt[:,:,::d,::d].cpu(), 'small', itm)
    write_images(gt.cpu(), 'xgt', itm)
    write_images(x.cpu(), 'x', itm)

    n_steps_each = args.iterations
    step_lr = args.lr
    t0 = time.time()
    for sigma, (checkpoint, step_multiplier) in checkpoints.items():
        load_part_of_model(model, os.path.join(args.load_params, checkpoint), prefix='')
    
        eta = step_lr * (sigma / .01)**2
        gamma = 1.0 / (sigma**2)
        print('sigma = {}'.format(sigma))
        for i, step in enumerate(range(n_steps_each*step_multiplier)):
            noise = torch.randn_like(x) * np.sqrt(eta*2)
            noise[:,:,::d,::d] = 0

            x = x.detach()
            x.requires_grad=True
            nll = discretized_mix_logistic_loss_smoothed_nll(x, model(x), sigma, h=64)
            grad_x = torch.autograd.grad(-nll.sum(), x)[0]
            grad_x[:,:,::d,::d] = 0

            x = x + eta * grad_x + noise

            norm = np.linalg.norm(grad_x.contiguous().view(-1,3*32*32).cpu().numpy(),axis=1).mean()

            smoothed_history.append(nll.mean().detach().cpu().numpy()/np.log(2.))

        pretrained = 'pcnn_lr.0.00040_nr-resnet5_nr-filters160_889.pth'
        load_part_of_model(model, os.path.join('pretrained', pretrained))

        with torch.no_grad(): xhat = model(x)
        unsmoothed_nll = discretized_mix_logistic_loss(x, xhat)/(np.prod(x.shape) * np.log(2.))

        recon = ((x - gt)**2).contiguous().view(-1,3*32*32).sum(1).mean()

        fmt = 'step: {}, recon: {}, nll: {}, smooth_nll: {}, |norm|: {}, time: {}'
        print(fmt.format(i,recon, unsmoothed_nll, nll.mean()/np.log(2),norm, time.time()-t0))
        t0 = time.time()

        write_images(x.detach().cpu(), 'x', itm)

        nll_history.append(unsmoothed_nll.detach().cpu().numpy())

    #print('test loss : %s' % evaluate(model, sigma)) 

np.save('opt/super_unsmoothed_nll_lr{}_T{}.npy'.format(step_lr, n_steps_each), nll_history)
np.save('opt/super_nll_lr{}_T{}.npy'.format(step_lr, n_steps_each), smoothed_history)
