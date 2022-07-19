# Testing FAtNet_v1 using S1 strategy

from dataloader import DataLoader
import utils
import models
import time
from tqdm import tqdm
import numpy as  np
from csv import writer
import pandas as pd   

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torchvision import models
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from utils import CenterLoss
import torch.nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm1d, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report, export_error_rates
from pyeer.plot import plot_eer_stats
from scipy import interpolate
from skimage.transform import resize


# Load data
print("Loading Data ... ")
data = DataLoader() 
test_x1, test_x2, test_y = data.librispeech_test_S3()#data.aishell1_test_S3() #data.multilingual_test_S3()#data.voxceleb1_test_S3() 
#test_x1 = np.array(test_x1)
#test_x2 = np.array(test_x2)
test_y = np.array(test_y)

# Batch Size
batch_size = 1
print("Batch Size = {}".format(batch_size))



# Load model
model = torch.load("/home/divyas/Workspace/Vox2Code/Code/Files/Results/FAtNet_vox2/Models/38__1613198443.9425883.pt")#60__1612722201.6210203_teacher.pt")#38__1613198443.9425883.pt")
model = model.cuda()

#https://github.com/zengchang94622/Speaker_Verification_Tencent/blob/master/inception_with_centloss.py
def eer(y_true, y_pred):
    fpr, tpr, thres = roc_curve(y_true, y_pred)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0, 1)
    thres = interp1d(fpr, thres)(eer)
    return eer, thres



def compute_metrics(output_prob, predictions, target):
    
    genuine_match_scores = []
    imposter_match_scores = []

    
    for i in tqdm(range(len(target))):
                    
        if target[i]==1:
            genuine_match_scores.append(output_prob[i])
        elif target[i]==0:
            imposter_match_scores.append(output_prob[i])
            
        total = total+1
        
    
    eer_value, threshold = eer(target, output_prob)
    
    print("Equal Error Rate (EER) : {}".format(eer_value))
    print("EER Threshold : {}".format(threshold))
    
  


probabilities = []
predictions = []

model.eval()
lst = [i for i in range(0, len(test_y))]
print(len(test_y))

for i in tqdm((range(0,len(test_y), batch_size))): 

    indices = lst[i:i+batch_size]

    batch_path_x1  = np.array(test_x1[i])
    batch_path_x2 = np.array(test_x2[i])


    batch_test_x1 = []
    for path in (batch_path_x1):
        mfcc1 = np.load(path)
        mfcc1 = mfcc1.astype(np.float32)
        mfcc1 = mfcc1.T
        batch_test_x1.append(mfcc1)

    batch_test_x1 = np.array(batch_test_x1)
    batch_test_x1 = torch.from_numpy(batch_test_x1)


    batch_test_x2 = []
    for path in (batch_path_x2):
        mfcc2 = np.load(path)
        mfcc2 = mfcc2.astype(np.float32)
        mfcc2 = mfcc2.T
        batch_test_x2.append(mfcc2)

    batch_test_x2 = np.array(batch_test_x2)
    batch_test_x2 = torch.from_numpy(batch_test_x2)

    batch_test_x1 = Variable(batch_test_x1)
    batch_test_x2 = Variable(batch_test_x2)

    #print("batch_test_x1 :",batch_test_x1.shape)
    #print("batch_test_x2 :",batch_test_x2.shape)

    with torch.no_grad( ) :
        embedding, output_prob, features = model(batch_test_x1.cuda(), batch_test_x2.cuda())
        embedding2, output_prob2, features2 = model(batch_test_x2.cuda(), batch_test_x1.cuda())


    softmax = torch.exp(output_prob.detach()).cpu()    
    prob = list(softmax.numpy()) #list(softmax.detach().cpu().numpy())

    softmax2 = torch.exp(output_prob2.detach()).cpu() 
    prob2 = list(softmax2.numpy())



    prob.extend(prob2)

    prob = [np.average(prob, axis = 0)]
    prob = np.array(prob)

    
    pred = np.argmax(prob, axis=1)


    probabilities.append(prob[0][1])
    predictions.append(pred)

    batch_test_x1.detach()
    batch_test_x2.detach()



    
    
    
print("PERFORMANCE ON TEST SET")    
compute_metrics(probabilities, predictions, test_y)



    
