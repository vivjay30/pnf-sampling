import torch

from train import build_model


# The various noise levels, geometrically spaced
CHECKPOINTS = {
     175.9         : 'checkpoints175pt9',
     110.          : 'checkpoints110pt0',
     68.7          : 'checkpoints68pt7',
     54.3          : 'checkpoints54pt3',
     42.9          : 'checkpoints42pt9',
     34.0          : 'checkpoints34pt0',
     26.8          : 'checkpoints26pt8',
     21.2          : 'checkpoints21pt2',
     16.8          : 'checkpoints16pt8',
     13.3          : 'checkpoints13pt3',
     10.5          : 'checkpoints10pt5',
     8.29          : 'checkpoints8pt29',
     6.55          : 'checkpoints6pt55',
     5.18          : 'checkpoints5pt18',
     4.1           : 'checkpoints4pt1',
     3.24          : 'checkpoints3pt24',
     2.56          : 'checkpoints2pt56',
     1.6           : 'checkpoints1pt6',
     1.0           : 'checkpoints1pt0',
     0.625         : 'checkpoints0pt625',
     0.39          : 'checkpoints0pt39',
     0.244         : 'checkpoints0pt244',
     0.15          : 'checkpoints0pt15',
     0.1           : 'checkpoints0pt0'
}


class ModelWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = build_model()
        self.model.eval()
        self.receptive_field = self.model.receptive_field

    def load_checkpoint(self, path):
        print("Load checkpoint from {}".format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["state_dict"])

    def forward(self, x, sigma, c=None):
        return self.model.smoothed_loss(x, c=c, sigma=sigma, batched=True)
