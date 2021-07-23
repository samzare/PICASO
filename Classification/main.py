import argparse
import os
import logging
import time
import torch.backends.cudnn as cudnn
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from data_smallNORB import ModelFetcher
from modules import *


class PiacsoModel(nn.Module):
    def __init__(
        self,
        dim_input=16,
        num_outputs=1,
        dim_output=5,
        num_inds=5,
        dim_hidden=128,
        num_heads=4,
        ln=False,
    ):
        super(PiacsoModel, self).__init__()
        self.enc = nn.Sequential(
            AE(dim_input, dim_hidden, num_heads, num_inds, ln=ln)
            AE(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PICASO(dim_hidden, num_heads, num_outputs, ln=ln),#, Vis=False),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        H = self.dec(self.enc(X))
        return H.squeeze()


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument("--num_pts", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--dim", type=int, default=256)
parser.add_argument("--n_heads", type=int, default=8)
parser.add_argument("--n_anc", type=int, default=1)
parser.add_argument("--train_epochs", type=int, default=501)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--save_freq', type=int, default=50)
parser.add_argument('--run_name', type=str, default='trial')


args = parser.parse_args()
args.exp_name = args.run_name
log_dir = "result/" + args.exp_name
model_path = log_dir + "/model"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'#,1,2,3'

generator = ModelFetcher(
    "./dataset/train_data.h5",
    "./dataset/test_data.h5",
    args.batch_size,
    down_sample=int(10000 / args.num_pts)
)

model = PiacsoModel(dim_hidden=args.dim, num_heads=args.n_heads, num_inds=args.n_anc)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()
model = nn.DataParallel(model)
model = model.cuda()

save_dir = os.path.join(log_dir)
def train():
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(args.run_name)
    logger.addHandler(logging.FileHandler(
        os.path.join(save_dir,
                     'train_' + time.strftime('%Y%m%d-%H%M') + '.log'),
        mode='w'))
    logger.info(str(args) + '\n')

    tick = time.time()
    for epoch in range(args.train_epochs):
        if epoch == int(0.5*args.train_epochs):
            optimizer.param_groups[0]['lr'] *= 0.1
        model.train()
        losses, total, correct = [], 0, 0
        for imgs, _, lbls in generator.train_data():
            imgs = torch.Tensor(imgs).cuda()
            lbls = torch.Tensor(lbls).long().cuda()
            preds = model(imgs)
            loss = criterion(preds, lbls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            total += lbls.shape[0]
            correct += (preds.argmax(dim=1) == lbls).sum().item()

        avg_loss, avg_acc = np.mean(losses), correct / total
        line = 'Epoch {}, '.format(epoch)
        line += 'train_loss {:.4f}, train_acc {:.4f}'.format(avg_loss, avg_acc)
        logger.info(line)

        if epoch % args.test_freq == 0:
            line = 'Epoch {}, '.format(epoch)
            line += test()
            line += ' ({:.3f} secs)'.format(time.time()-tick)
            tick = time.time()
            logger.info(line)

        if epoch % args.save_freq == 0:
            torch.save({'state_dict':model.state_dict()},
                    os.path.join(save_dir, 'model.tar'))

    torch.save({'state_dict':model.state_dict()},
        os.path.join(save_dir, 'model.tar'))

def test():
    model.eval()
    losses, total, correct = [], 0, 0
    for imgs, _, lbls in generator.test_data():
        imgs = torch.Tensor(imgs).cuda()
        lbls = torch.Tensor(lbls).long().cuda()
        preds = model(imgs)
        loss = criterion(preds, lbls)

        losses.append(loss.item())
        total += lbls.shape[0]
        correct += (preds.argmax(dim=1) == lbls).sum().item()
    avg_loss, avg_acc = np.mean(losses), correct / total

    line = 'test_loss {:.4f}, test_acc {:.4f}'.format(avg_loss, avg_acc)

    return line


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        ckpt = torch.load(os.path.join(save_dir, 'elevation32.tar'))
        model.load_state_dict(ckpt['state_dict'])
        test()
