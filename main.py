import numpy as np
import cv2
import os
import json
import torch
import torch.nn as nn
from math import sin, cos, pi, sqrt
from time import time
import wget
import zipfile

if not os.path.exists('train_dataset_train.zip'):
    print("Грузим train")
    url1 = 'https://lodmedia.hb.bizmrg.com/case_files/768820/train_dataset_train.zip'
    path = '.'
    wget.download(url1, out=path)

if not os.path.exists('test_dataset_test.zip'):
    print("Грузим test")
    url2 = 'https://lodmedia.hb.bizmrg.com/case_files/768820/test_dataset_test.zip'
    path = '.'
    wget.download(url2, out=path)

# if not os.path.isdir('train'):
#     os.mkdir('train')
#     # os.mkdir('train/json')
#     # os.mkdir('train/test')

if not os.path.isdir('test'):
    os.mkdir('test')
    os.mkdir('test/img')

if not os.path.isdir('submit'):
    os.mkdir('submit')

print("Разархивируем train")
with zipfile.ZipFile('train_dataset_train.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

print("Разархивируем test")
with zipfile.ZipFile('test_dataset_test.zip', 'r') as zip_ref:
    zip_ref.extractall('test/img')

rf = 4  # resize factor (1, 2, 4)
step = 4
astep = 3

h = 1024
H = 10496
h4 = h // rf
H4 = W4 = H // rf
print(h4, H4)
nk = 400
train = False


def get_pad():
    image = cv2.imread("original.tiff")
    pad0 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pad0 = cv2.resize(pad0, (W4, H4))
    m, s = pad0.mean().astype(np.float32), pad0.std().astype(np.float32)
    return (pad0 - m) / s


def get_sample(id, train=False):
    dir = 'train' if train else 'test'
    image = cv2.imread(f"{dir}/img/{id}.png")
    sample = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sample = cv2.resize(sample, (h4, h4))
    m, s = sample.mean().astype(np.float32), sample.std().astype(np.float32)
    return (sample - m) / s


# def get_params(id, train=False):
#     dir = 'train' if train else 'test'
#     return json.load(open(f"{dir}/json/{id}.json"))


def rotate_pad(angle: int, pad=None):
    a = pi*angle/180
    si_, co_ = sin(a), cos(a)
    X0, Y0, L = getX0Y0(a, si_, co_)
    if pad is None:
        pad = get_pad()
    rot_pad = np.zeros((L, L), dtype=np.float32)
    for Y in range(L):
        Y1 = Y - Y0
        Ys = Y1 * si_
        Yc = Y1 * co_
        for X in range(L):
            X1 = X - X0
            x = int(X1 * co_ - Ys)
            if x < 0 or x >= H4:
                continue
            y = int(X1 * si_ + Yc)
            if y < 0 or y >= H4:
                continue
            rot_pad[Y, X] = pad[y, x]
    return rot_pad


def getX0Y0(a, si_, co_):
    if a < pi/2:
        X0 = 0
        Y0 = si_
        L = sqrt(2) * sin(a + pi/4)
    elif a < pi:
        X0 = -co_
        Y0 = sqrt(2) * sin(a - pi/4)
        L = Y0
    elif a < 1.5 * pi:
        X0 = sqrt(2) * sin(a - 3*pi/4)
        Y0 = -co_
        L = X0
    else:
        X0 = -si_
        Y0 = 0
        L = sqrt(2) * cos(a + pi/4)
    return [int(H4 * X0), int(H4 * Y0), int(H4 * L)]


def get_XY(x, y, angle):
    a = pi*angle/180
    si_, co_ = sin(a), cos(a)
    X0, Y0, _ = getX0Y0(a, si_, co_)
    X = X0 + int(x * co_ + y * si_)
    Y = Y0 + int(y * co_ - x * si_)
    return [X, Y]


def get_xy(X, Y, angle):
    a = pi*angle/180
    si_, co_ = sin(a), cos(a)
    X0, Y0, _ = getX0Y0(a, si_, co_)
    x = int((X - X0) * co_ - (Y - Y0) * si_)
    y = int((Y - Y0) * co_ + (X - X0) * si_)
    return [rf * x, rf * y]


class MyModel(nn.Module):
    def __init__(self, n=400, step=20):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(1, n, (h4, h4), (step, step))

    def forward(self, x):
        x = self.conv(x)
        return x


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Device: {}'.format(device))


dir = 'train' if train else 'test'
img_dir = dir + "/img/"

filenames = sorted(os.listdir(img_dir))
ids = [x.split('.')[0] for x in filenames if x.endswith('png')]

if train:
    ids = ids[:nk]

n = len(ids)
print(f"Creating model n={n}, step={step}")

model = MyModel(n=n, step=step)

print(f"Initializing model weights")
for i, id in enumerate(ids):
    snap = get_sample(id, train=train)
    model.conv.weight.data[i, 0] = torch.from_numpy(snap)
model.eval()
model = model.to(device)

""" Main cycle - create json files in the `sumbit` folder"""
scores = {}
results = {}
for i, id in enumerate(ids):
    scores[id] = 0.
    results[id] = {}
since = time()
pad0 = get_pad()
print("Making pad0 took %.1f sec" % (time() - since))
for angle in range(0, 360, astep):
    since = time()
    pad = rotate_pad(angle, pad0)
    print("Angle=%d Making pad took %.1f sec" % (angle, time() - since))
    pad = torch.from_numpy(pad).unsqueeze_(0)
    padc = pad.to(device)
    since = time()
    with torch.no_grad():
        out = model(padc)
    print("Inference took %.3f sec" % (time() - since))
    since = time()
    Ny, Nx = out.shape[1:]
    print(f"Ny={Ny}, Nx={Nx}")
    for k, id in enumerate(ids):
        dat1 = out[k]
        maxind = torch.argmax(dat1).item()
        i = maxind // Ny
        j = maxind % Ny
        val = dat1[i, j] / h4 / h4
        if val > scores[id]:
            scores[id] = val
            bestX, bestY = j*step, i*step
            results[id]['left_top'] = get_xy(bestX, bestY, angle)
            results[id]['right_top'] = get_xy(bestX + h4, bestY, angle)
            results[id]['left_bottom'] = get_xy(bestX, bestY + h4, angle)
            results[id]['right_bottom'] = get_xy(bestX + h4, bestY + h4, angle)
            results[id]['angle'] = angle
    print("Search took %.3f sec" % (time() - since))
    print("------------------------------")
for id in ids:
    print("id=%s Score: %.3f" % (id, scores[id]), results[id])
    filename = f"submit/{id}.json"
    with open(filename, "w") as f:
        json.dump(results[id], f)
