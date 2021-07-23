import argparse
import os
import logging
import time
import torch.backends.cudnn as cudnn
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score, auc
from data_celebA import ModelFetcher
from modules import *
#from classifier import *

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument("--num_pts", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--dim", type=int, default=256) #256
parser.add_argument("--n_heads", type=int, default=8) #8
parser.add_argument("--n_anc", type=int, default=8)
parser.add_argument("--train_epochs", type=int, default=501)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--save_freq', type=int, default=50)
parser.add_argument('--run_name', type=str, default='trial_celebA')


args = parser.parse_args()

class PicasoModel(nn.Module):
    def __init__(
        self,
        dim_input=256, #576,
        num_outputs=1,
        dim_output=8,
        num_inds=1,
        dim_hidden=args.dim,
        num_heads=args.n_heads,
        ln=False,
    ):
        super(PicasoModel, self).__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ELU(),
            #nn.Dropout(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=5, stride=2),
            #nn.Dropout(p=0.2),
        )
        self.lin = nn.Sequential(
            nn.Linear(256, dim_input),
            #nn.Linear(512, 128),
        )
        self.enc = nn.Sequential(
            nn.Dropout(),
            SA(dim_input, dim_hidden, num_heads, ln=ln),
            SA(dim_hidden, dim_hidden, num_heads, ln=ln),
            SA(dim_hidden, dim_hidden, num_heads, ln=ln),
            SA(dim_hidden, dim_hidden, num_heads, ln=ln),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            #PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            #PICASO(dim_hidden, dim_hidden, num_heads, num_outputs, ln=ln),
            Gen_PICASO(dim_hidden, dim_hidden, num_heads, num_outputs, ln=ln),

            nn.Dropout(),
            nn.Linear(dim_hidden, 1),
            nn.Softmax(dim=1)
        )


    def forward(self, X):
        B, N, C, H, W = X.shape
        X_flat = X.view(B * N, C, H, W)
        prep = (self.prep(X_flat).view(B, N, -1)) 
        H_enc = self.dec(self.enc(prep)) 
        return (H_enc).squeeze(2)



args.exp_name = args.run_name
log_dir = "result/" + args.exp_name
model_path = log_dir + "/model"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

generator = ModelFetcher(
    "train.npz",
    "test.npz",
    args.batch_size,
    down_sample=int(10000 / args.num_pts)
)

model = PicasoModel(dim_hidden=args.dim, num_heads=args.n_heads, num_inds=args.n_anc)
#model = DeepSet(network_dim=256)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = nn.BCEWithLogitsLoss()
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
            lbls = torch.Tensor(lbls).float().cuda()
            preds = model(imgs)
            loss = criterion(preds, lbls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            total += lbls.shape[0]
            correct += (preds.argmax(dim=1) == lbls.argmax(dim=1)).sum().item() #(preds.argmax(dim=1) == lbls).sum().item()

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
    losses, total, correct,  = [], 0, 0
    precision = dict()
    recall = dict()
    average_precision = dict()
    targets, score = [], []
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for imgs, lbls in generator.test_data():
        imgs = torch.Tensor(imgs).cuda()
        lbls = torch.Tensor(lbls).float().cuda()
        targets += [lbls]
        preds = model(imgs)
        score += [preds]
        loss = criterion(preds, lbls) #criterion(torch.max(pred, 1)[1], torch.max(lbls, 1)[1])

        losses.append(loss.item())
        total += lbls.shape[0]
        correct += (preds.argmax(dim=1) == lbls.argmax(dim=1)).sum().item()  #(preds.argmax(dim=1) == lbls).sum().item()


    avg_loss, avg_acc = np.mean(losses), correct / total
    targets = torch.cat(targets)
    score = torch.cat(score)

    '''for i in range(8):
        precision[i], recall[i], _ = precision_recall_curve(targets[:, i].cpu().detach().numpy(),
                                                            score[:, i].cpu().detach().numpy())
        average_precision[i] = average_precision_score(targets[:, i].cpu().detach().numpy(),
                                                       score[:, i].cpu().detach().numpy())

        fpr[i], tpr[i], _ = roc_curve(targets[:, i].cpu().detach().numpy(),
                                                            score[:, i].cpu().detach().numpy())
        roc_auc[i] = auc(fpr[i], tpr[i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(targets.cpu().detach().numpy().ravel(),
                                                                        score.cpu().detach().numpy().ravel())
    average_precision["micro"] = average_precision_score(targets.cpu().detach().numpy(), score.cpu().detach().numpy(),
                                                             average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
              .format(average_precision["micro"]))

    aupr = auc(recall["micro"], precision["micro"])
    print(aupr)

    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision["micro"]))
    plt.show()

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(targets.cpu().detach().numpy().ravel(),
                                                                        score.cpu().detach().numpy().ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print(roc_auc["micro"])
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()'''

    line = 'test_loss {:.4f}, test_acc {:.4f}'.format(avg_loss, avg_acc)
    print(line)
    return line

if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        ckpt = torch.load(os.path.join(save_dir, 'model.tar'))
        model.load_state_dict(ckpt['state_dict'])
        test()
