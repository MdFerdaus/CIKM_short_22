import matplotlib
#matplotlib.use('Agg')
import argparse
import os
import numpy as np
import random
import torch.nn as nn
# import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import classification_report
import copy
import random as ran
import torch.nn.functional
from utils import  ComplementEntropy, indices, one_hot_encode,  indices, calc_gradient_penalty, _biased_sample_labels
####, get_test_metrics, calc_gradient_penalty, _set_class_ratios, get_minor_class
from torchvision import transforms, utils, datasets
from models_gumbel import gen_maps, weights_init, MLP_classifiers, dis_wgan, one_hot_encode
from torchvision.utils import save_image
# from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from matplotlib import pyplot
from numpy import where
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import random as r
from tensorboardX import SummaryWriter
import sys
import statistics
from sklearn.preprocessing import LabelEncoder

import math
writer = SummaryWriter()

manualSeed = 3964
seed = manualSeed


# If you are using CUDA on 1 GPU, seed it
torch.cuda.manual_seed(0)
# # If you are using CUDA on more than 1 GPU, seed them all
torch.cuda.manual_seed_all(0)
# Disable the inbuilt cudnn auto-tuner that finds the best algorithm to use for your hardware.
torch.backends.cudnn.benchmark = False
# Certain operations in Cudnn are not deterministic, and this line will force them to behave!
torch.backends.cudnn.deterministic = True








#### datasets preparation here
#### datasets defined here
full_data = pd.read_csv("data_200NM.csv")
data = full_data
test_nohold = data[data["delay_2"] == 0].sample(n=960, random_state=0)  #   holding
test_hold = data[data["delay_2"] == 1].sample(n=960, random_state=0)
train_nohold = data[data["delay_2"] == 0].drop(test_nohold.index)
train_hold = data[data["delay_2"] == 1].drop(test_hold.index)

test_set = pd.concat([test_nohold, test_hold])
train_set = pd.concat([train_nohold, train_hold])

feature =['lat', 'long', 'heading', 'altitude','groundspeed',
         'average_spd', 'DOW', 'HOD', 'wake',
         'bef_30', 'aft_30','TAF_idx','region','runway',
         'arr_rwy_02L_cap', 'arr_rwy_20R_cap', 'arr_rwy_02C_cap','arr_rwy_20C_cap',
         'dep_rwy_02L_cap', 'dep_rwy_20R_cap', 'dep_rwy_02C_cap','dep_rwy_20C_cap', 'STAR_grp', 'holding']

X_test = test_set[feature]
y_test = test_set['delay_2']

X_train = train_set[feature]
y_train = train_set['delay_2']

#### Converting categorical data into numerical ======
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
### End of definition

X_train = MultiColumnLabelEncoder(columns=['region', 'runway', 'wake']).fit_transform(X_train)
X_test = MultiColumnLabelEncoder(columns=['region', 'runway', 'wake']).fit_transform(X_test)

X_train = X_train.to_numpy()
y_train = y_train.to_numpy()

X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

print([np.count_nonzero(y_train == c) for c in range(2)])

#### shuffle data
def _shuffle(x, y, seed):
    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(seed)
    np.random.shuffle(y)
    return x, y



############### oversampling code here

def oversampling(aug_rate, x, y, seed):   #### aug_rate = 1
    n_classes = len(np.unique(y))
    #print(n_classes)
    class_cnt = [np.count_nonzero(y == c) for c in range(n_classes)]
    #print("class_cnt", class_cnt)
    max_class_cnt = max(class_cnt)
    x_aug_list = []
    y_aug_list = []
    if aug_rate <= 0:
       return x, y
    aug_nums = [aug_rate * (max_class_cnt - class_cnt[i]) for i in range(n_classes)]
    rep_nums = [aug_num / class_cnt[i] for i, aug_num in enumerate(aug_nums)]
    for i in range(n_classes):
        idx = (y == i)
        if rep_nums[i] <= 0.:
           x_aug_list.append(x[idx])
           y_aug_list.append(y[idx])
           continue
        n_c = np.count_nonzero(idx)
        if n_c == 0:
            continue
        x_aug_list.append(
                np.repeat(x[idx], repeats=math.ceil(1 + rep_nums[i]), axis=0)[:math.floor(n_c * (1 + rep_nums[i]))])
        y_aug_list.append(
                np.repeat(y[idx], repeats=math.ceil(1 + rep_nums[i]), axis=0)[:math.floor(n_c * (1 + rep_nums[i]))])
        if len(x_aug_list) == 0:
            return x, y
    x_aug = np.concatenate(x_aug_list, axis=0)
    y_aug = np.concatenate(y_aug_list, axis=0)
    x_aug, y_aug = _shuffle(x_aug, y_aug, seed)
    print([np.count_nonzero(y_aug == c) for c in range(n_classes)])
    return x_aug, y_aug

##### oversample data here
aug_rate = 0
x_aug, y_aug = oversampling(aug_rate, X_train, y_train, seed)

#oversample = SMOTE()
#oversample = ADASYN()
#oversample = BorderlineSMOTE()

#x_aug, y_aug = oversample.fit_resample(X_train, y_train)


assert len(x_aug) == len(y_aug)

print(x_aug.shape)
print(y_aug.shape)

counter_aug = Counter(y_aug)
print(counter_aug)
counter_train = Counter(y_train)
print(counter_train)
counter_test = Counter(y_test)
print(counter_test)



aug_data = torch.Tensor(x_aug).float()  # transform to torch tensor
aug_label = torch.Tensor(y_aug).long()

train_data = torch.Tensor(x_aug).float() # transform to torch tensor
train_label = torch.Tensor(y_aug).long()
test_data_ = torch.Tensor(X_test).float() # transform to torch tensor
test_label_ = torch.Tensor( y_test).long()

train_dataset = TensorDataset(train_data,train_label) # create your datset
test_dataset = TensorDataset(test_data_,test_label_) # create your datset
print(len(train_dataset))
aug_dataset = TensorDataset(aug_data, aug_label)  # create your datset



## model parametrs define here

batch_size = 64
base_lr = 0.00001
beta1 = 0.5
input_dim = 24  #### datashape[1]
num_class = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decay =  2.5*1e-5   #### working as a regulisers
cuda = True if torch.cuda.is_available() else False
if device:
    print("the machine has a device")

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=2)

####GAMO generator defines here
balanced_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=2)

ao_data, ao_class = iter(balanced_loader).next()
test_data, test_labels = iter(test_loader).next()

if cuda:
    ao_data = Variable(ao_data.type(Tensor))
    ao_class = ao_class.cuda()
    test_data = Variable(test_data.type(Tensor))
    test_labels = test_labels.cuda()



### from pakdd ao
_, per_class_count = np.unique(train_label, return_counts=True)

print(per_class_count[0])

toBalance = np.subtract(per_class_count[0], per_class_count[1:])

print(toBalance)

def randomLabelGen(toBalance, batchSize):
    cumProb=np.cumsum(toBalance/np.sum(toBalance))
    bins=np.insert(cumProb, 0, 0)
    randomValue=np.random.rand(batchSize,)
    randLabel=np.digitize(randomValue, bins)

    return randLabel



# # ################# model defines here
best_acc = 0  # best test accuracy
best_gm = 0
best_tpr = 0
count2 = 0
count1 = 0

epochs = 301
lambda_cons = 1
lambda_cot = 1


tpr_max  = 0
roc_score_max = 0

#### train main models here
iteration = 0
count = 1
n_skip_iter = 1
nclasses = 2
real_adv_label = 1
fake_adv_label = 0

genrated_sample = int((batch_size)/nclasses)


# ##### take n-1 samples from  minor classes


# Tensor = torch.FloatTensor
all_max_GM=[]
avg_GM=[]

all_max_ACSA=[]
avg_ACSA=[]

for iter in range(3):
    ############ model defines here
    gen = gen_maps(ao_data, ao_class, n_classes=num_class)
    mlp_class = MLP_classifiers(imSize=input_dim, n_class=num_class)
    gen.apply(weights_init)
    mlp_class.apply(weights_init)

    #### loss defines here
    CE_loss = nn.CrossEntropyLoss()
    ##### complementrary loss here
    compl_loss = ComplementEntropy(n_c=num_class)
    bce_loss = nn.BCELoss()

    if cuda:
        gen.cuda()
        mlp_class.cuda()
        # dis.cuda()
        CE_loss.cuda()
        compl_loss.cuda()
        bce_loss.cuda()

    optim_comp = optim.Adam(gen.parameters(), lr=base_lr, betas=(0.5, 0.9999))

    ########
    optim_class = optim.Adam(mlp_class.parameters(), lr=base_lr, betas=(0.5, 0.9999))

    acsa_max = 0
    gm_max = 0

    for epoch in range(epochs):
        i = 0
        print("epoch", epoch)
        # Ensure generator/encoder are trainable
        for i, (data, itruth_label) in enumerate(train_loader):
            gen.train()
            mlp_class.train()

         ##### update classifier

            optim_class.zero_grad()
         #optim_coss.zero_grad()

            if cuda:
                real_data = Variable(data.type(Tensor))
                real_label = itruth_label.cuda()

         #real_data = real_data.view(-1,784)
            batch_size = data.size()[0]
            real_class_score = mlp_class(real_data)
            real_class_loss = CE_loss(real_class_score, real_label)

            real_class_loss.backward()
            optim_class.step()

        # ###### generated samples for fake data
            optim_class.zero_grad()
            gen.eval()
            augment_samples = randomLabelGen(toBalance, batch_size)   #### oversample is for different structure
            augment_samples = torch.from_numpy(augment_samples).long().cuda()   #### minority oversample is for different structure
            zc_aug = one_hot_encode(augment_samples)
            zn_aug = Variable(torch.rand(batch_size, 100).type(Tensor))
            augmented_feature = gen(zn_aug,zc_aug, augment_samples)   #####
            aug_class_score = mlp_class(augmented_feature)
            aug_ce_loss = CE_loss(aug_class_score, augment_samples)


            aug_ce_loss.backward()
            optim_class.step()


        ##### generator training here
            if i%n_skip_iter ==0:
                gen.train()
                optim_comp.zero_grad()


                balanced_generate_count =  genrated_sample
                generate_label = torch.from_numpy(np.arange(0,nclasses).repeat(balanced_generate_count)).long().cuda()
                zn_fake = Variable(torch.rand(generate_label.size()[0], 100).type(Tensor))
                zc_fake = one_hot_encode(generate_label)
                fake_image = gen(zn_fake,zc_fake, generate_label)   #####

                gen_feat_score = mlp_class(fake_image)
                comp_loss = CE_loss(gen_feat_score,generate_label)   #### crossentropy loss only
                comp_loss.backward()
                optim_comp.step()

                count2 +=1

        if epoch%1 ==0:
            mlp_class.eval()
             #model_feat.eval()
            pred_score = mlp_class(test_data)
             #original_pred_score = mlp_class(original_pred_feat)
            pred_score = pred_score.data.cpu().numpy()
            y_pred = np.argmax(pred_score, axis=1)
            gt = test_labels.data.cpu().numpy()

            acsa, gm, tpr, confMat, acc =  indices(y_pred, gt)
             #print("tpr", tpr)
            print("gm", gm)
             #print("acc", acc)
            print("acsa", acsa)
            print("confMat", confMat)
            roc_score = roc_auc_score(gt, y_pred, multi_class='ovr')

            if roc_score > roc_score_max :
                roc_score_max = roc_score
            if acsa > acsa_max :
                acsa_max = acsa
                gm_max = gm

            mlp_class.train()

    print("Max_ACSA", acsa_max)
    print("Max_GM", gm_max)
    all_max_ACSA.append(acsa_max)
    all_max_GM.append(gm_max)

print("all_max_ACSA", all_max_ACSA)
print("all_max_GM", all_max_GM)

avg_ACSA = sum(all_max_ACSA)/len(all_max_ACSA)
std_ACSA=statistics.stdev(all_max_ACSA)
print("avg_ACSA", avg_ACSA)
print("std_ACSA", std_ACSA)

avg_GM = sum(all_max_GM)/len(all_max_GM)
std_GM=statistics.stdev(all_max_GM)
print("avg_GM", avg_GM)
print("std_GM", std_GM)



