import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from torch.nn.parallel import data_parallel


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0.00)


def one_hot_encode(labels, nc=2, cuda=True):
    nbatch = labels.size()[0]
    if cuda:
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor
    if cuda:
        # Pure one-hot vector generation
        zc_FT = Tensor(nbatch, nc).fill_(0)
        zc_FT = zc_FT.scatter_(1, labels.unsqueeze(1), 1.)
    else:
        zc_FT = Tensor(nbatch, nc).fill_(0)
        zc_FT = zc_FT.scatter_(1, labels.unsqueeze(1), 1.)
    return zc_FT


def sample_gumbel(shape, eps=1e-3):
    U = eps * torch.rand(shape).cuda()
    # print("U", U)
    return -Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


class single_generator_block(nn.Module):
    def __init__(self, per_class_feat_data, nz=100, n_c=2):
        super(single_generator_block, self).__init__()
        self.per_class_feat_data = per_class_feat_data
        self.feat_number = per_class_feat_data.size()[0]
        self.nz = nz
        self.nc = n_c
        self.temp = 1
        self.feature_dim = self.per_class_feat_data.size()[1]
        self.feature_latent = nn.Sequential(
            nn.Linear(self.nz + self.nc, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, self.feat_number))

    def forward(self, x, y):  ##### x input data
        z = torch.cat((x, y), -1)
        z = self.feature_latent(z)
        z = gumbel_softmax(z, 0.1)
        z = z.unsqueeze(2).repeat(1, 1, self.feature_dim)
        output = torch.sum(z * self.per_class_feat_data, dim=1)  ### batch, image_shape
        output = output.view(-1, self.feature_dim)

        return output


class gen_maps(nn.Module):
    def __init__(self, feat_data, class_map, n_classes=2):
        super(gen_maps, self).__init__()
        self.feat_data = feat_data
        self.class_map = class_map
        self.classes = n_classes
        #### #generator block here

        self.all_layers = nn.ModuleList()

        for i in range(self.classes):  ##### all the generators have to be activated
            layer = []
            mask = self.class_map == i
            per_class_feature = self.feat_data[mask]  #### data prior selected
            layer.append(single_generator_block(per_class_feature))
            self.all_layers.append(*layer)

    def forward(self, zn, zc, batch_labels):
        latent_code = []
        indices = torch.unique(batch_labels, dim=0)  ####selecting the batch_labels for identity
        for i in indices:
            select_mask = batch_labels == i
            latent_output = self.all_layers[i](zn[select_mask], zc[select_mask])
            latent_code.append(latent_output)

        gen_image = torch.cat(latent_code, dim=0)
        return gen_image


##################  tabular data generators here
#### CGAN structure here

class gamo_dis(nn.Module):
    def __init__(self, input_dim=784, n_c=10):
        super(gamo_dis, self).__init__()
        self.features = input_dim
        self.nc = n_c
        self.main = nn.Sequential(
            nn.Linear(self.features + self.nc, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid())

    def forward(self, x, zc):
        # x = x.view(-1,784)
        z = torch.cat((x, zc), dim=-1)

        output = self.main(z)
        return output.squeeze()


class MLP_classifiers(nn.Module):
    def __init__(self, imSize=30, n_class=2):
        super(MLP_classifiers, self).__init__()
        self.imSize = imSize
        self.n_class = n_class
        self.class_score = nn.Sequential(
            nn.Linear(self.imSize, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2, inplace=False),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2, inplace=False),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2, inplace=False),
            nn.Linear(64, self.n_class))

    def forward(self, z):
        class_score = self.class_score(z)
        return class_score


class MLP_generator(nn.Module):
    def __init__(self, feat_data=100, n_c=10):  #### n_c = one hot code
        super(MLP_generator, self).__init__()
        self.features = feat_data
        self.n_c = n_c
        self.classf = nn.Sequential(
            nn.Linear(self.features + self.n_c, 512),
            nn.LeakyReLU(0.1, inplace=True),  ### nn.Tanh(),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, self.n_class)
        )

    def forward(self, x):
        class_score = self.classf(x)
        return class_score


##### fature discriminator for WGAN-GP
class dis_wgan(nn.Module):
    def __init__(self, feat_data=512):
        super(dis_wgan, self).__init__()
        self.features = feat_data
        self.main = nn.Sequential(
            nn.Linear(self.features, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1))

    def forward(self, x):
        output = self.main(x)
        return output

