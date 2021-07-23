from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import csv
import numpy as np

file = './celebA/Anno/list_attr_celeba.txt'

dataset = []
sets = {}
selected_file = []
anomaly_file = []
non_anom = []
attr2idx = {}
idx2attr = {}

lines = [line.rstrip() for line in open(file, 'r')]
all_attr_names = lines[1].split()
for i, attr_name in enumerate(all_attr_names):
    attr2idx[attr_name] = i
    idx2attr[i] = attr_name

lines = lines[2:]
random.seed(1234)
random.shuffle(lines)
for counter in range(18000):
    dataset = []
    non_anom = []
    selected_file = []
    anomaly_file = []

    selected_attrs = np.random.randint(0, 39, (2,))
    for i, line in enumerate(lines):
        split = line.split()
        filename = split[0]
        values = split[1:]

        label = []
        for attr_name in selected_attrs:
            idx = attr_name
            label.append(values[idx] == '1')


        if label == [True, True]:
            non_anom.append([filename, label])

        if label == [False, False]:
            anomaly_file.append([filename, label])

    if len(non_anom) >= 15:
        for k in range(15):
            idd = np.random.randint(0, len(non_anom), 1)
            dataset.append(non_anom[idd[0]])

    idd = np.random.randint(0, len(anomaly_file), 1)
    dataset.append(anomaly_file[idd[0]])
    dataset.append(selected_attrs)

    sets[counter] = dataset
    #if (i+1) < 2000:
        #test_dataset.append([filename, label])
    #else:
        #train_dataset.append([filename, label])

print('Finished preprocessing the CelebA dataset...')
w = csv.writer(open("output.csv", "w"))
for key, val in sets.items():
    w.writerow([key, val])

file = './output.csv'
with open(file, mode='r') as infile:
    reader = csv.reader(infile)
    with open('coors_new.csv', mode='w') as outfile:
        #writer = csv.writer(outfile)
        mydict = {rows[0]:rows[1] for rows in reader}

X = []
Y = []
Attrs = []
for key, val in mydict.items():
    if len(val) > 50:
        x = []
        split = val.split()
        attr = [split[-2][-2], split[-1][1]]
        if attr[0] != attr[1]:
            x.append(split[0][3:10]+'png')
            x.append(split[3][2:9] + 'png')
            x.append(split[6][2:9] + 'png')
            x.append(split[9][2:9] + 'png')
            x.append(split[12][2:9] + 'png')
            x.append(split[15][2:9] + 'png')
            x.append(split[18][2:9] + 'png')
            '''x.append(split[21][2:9] + 'png')
            x.append(split[24][2:9] + 'png')
            x.append(split[27][2:9] + 'png')
            x.append(split[30][2:9] + 'png')
            x.append(split[33][2:9] + 'png')
            x.append(split[36][2:9] + 'png')
            x.append(split[39][2:9] + 'png')
            x.append(split[42][2:9] + 'png')'''
            x.append(split[45][2:9] + 'png')
            y = [0,0,0,0,0,0,0,1]
            X.append(x)
            Y.append(y)
            Attrs.append(attr)

X_train = []
Y_train = []
X_test = []
Y_test = []
Att_train = []
Att_test = []
for i in range(1600):
    if i<1200:
        X_train.append(X[i])
        Y_train.append(Y[i])
        Att_train.append(Attrs[i])
    else:
        X_test.append(X[i])
        Y_test.append(Y[i])
        Att_test.append(Attrs[i])

# save

np.savez_compressed('train', data=X_train, label=Y_train)
np.savez_compressed('test', data=X_test, label=Y_test)


# load
train_arr = np.load('train.npz')['data']
#test_arr = np.load('train.npz')['data']
print('Finished')
