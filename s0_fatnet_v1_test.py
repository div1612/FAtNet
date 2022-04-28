# Testing StAtNet on VoxCeleb 1-test set using S0 strategy

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
#from skimage.transform import resize
import FAtNetModel

# Load data
print("Loading Data ... ")
data = DataLoader() 
test_x1, test_x2, test_y = data.voxceleb1_test_S0()#data.language_test_S0()#data.multilingual_test_S0()#data.aishell_multilingual_test_S0()#data.librispeech_multilingual_test_S0()#data.multilingual_test_S0()#data.multilingual_test_S0()#data.voxceleb1_test_S0()#data.voxceleb1_test_S0()#data.voxceleb1_test_S#for Low threshold 
test_x1 = np.array(test_x1)
test_x2 = np.array(test_x2)
test_y = np.array(test_y)

# Batch Size
batch_size = 1
print("Batch Size = {}".format(batch_size))



# Load model
#model = torch.load("/home/divyas/Workspace/AT/Vox2Code/Code/Files/Results/FAtNet_vox2/Models/38__1613198443.9425883.pt")#torch.load("/home/divyas/Workspace/AT/Vox2Code/Code/Files/Results/StAtNetAttnv6/Models/60__1612722201.6210203_teacher.pt")#60__1612722201.6210203_teacher.pt")#38__1613198443.9425883.pt")
#model = torch.load("/home/divyas/Workspace/AT/Vox2Code/Code/Files/Results/FAtNetv1_vox2_iui/Models/32__1631679306.5873108.pt")
model = torch.load("/home/divyas/Workspace/AT/Vox2Code/Code/Files/Results/FAtNet_vox2/FAtNetVox2epoch38.pt")#
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
    
    accuracy = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    precision = 0
    recall = 0
    f1 = 0 
    total = 0
    
    
    for i in tqdm(range(len(target))):
        
        if predictions[i]==target[i]:
            accuracy = accuracy+1
            
        if target[i]==1 and predictions[i]==1:
            tp = tp+1
            
        if target[i]==1 and predictions[i]==0:
            fn = fn+1
            
        if target[i]==0 and predictions[i]==0:
            tn = tn+1
            
        if target[i]==0 and predictions[i]==1:
            fp = fp+1
            
        if target[i]==1:
            genuine_match_scores.append(output_prob[i])
        elif target[i]==0:
            imposter_match_scores.append(output_prob[i])
            
        total = total+1
        
    accuracy = float(accuracy)/float(total)
    precision = float(tp)/( float(tp) + float(fp) )
    recall = float(tp)/( float(tp) + float(fn) )
    f1 = 2*precision*recall/(precision+recall)
    
    eer_value, threshold = eer(target, output_prob)
    

    
    print("True Positives : {}".format(tp))
    print("True Negatives : {}".format(tn))
    print("False Positives : {}".format(fp))
    print("False Negatives : {}".format(fn))
    print("Accuracy : {}".format(accuracy))
    print("Precision : {}".format(precision))
    print("Recall : {}".format(recall))
    print("F1 - Score : {}".format(f1))
    print("Equal Error Rate (EER) : {}".format(eer_value))
    print("EER Threshold : {}".format(threshold))
    '''
    stats = get_eer_stats(genuine_match_scores, imposter_match_scores)
    generate_eer_report([stats], ['Testing'],'/home/divyas/Workspace/AT/Vox2Code/Code/Res/Speech_FAtNetVox2Train_Turkish_Test_S5/S5_FAtNet_epoch38_Vox_eer_report.csv')
    export_error_rates(stats.fmr, stats.fnmr, '/home/divyas/Workspace/AT/Vox2Code/Code/Res/Speech_FAtNetVox2Train_Turkish_Test_S5/S5_FAtNet_epoch38Vox_Test_DET.csv')
    plot_eer_stats([stats], ['Testing'])

    
    with open('/home/divyas/Workspace/AT/Vox2Code/Code/Res/Speech_FAtNetVox2Train_Turkish_Test_S5/S5_FAtNet_epoch38_Test_genuine_match_scores.txt', mode='wt') as gscore:
        gscore.write('\n'.join(str(line) for line in genuine_match_scores))

    with open('/home/divyas/Workspace/AT/Vox2Code/Code/Res/Speech_FAtNetVox2Train_Turkish_Test_S5/S5_FAtNet_epoch38_Test_imposter_match_scores.txt', mode='wt') as iscore:
        iscore.write('\n'.join(str(line) for line in imposter_match_scores))
    '''

    
    

probabilities = []
predictions = []

model.eval()
lst = [i for i in range(0, len(test_y))]
print(len(test_y))

for i in tqdm((range(0,len(test_y), batch_size))): 
    
    indices = lst[i:i+batch_size]
    batch_path_x1  = test_x1[indices] 
    batch_path_x2 = test_x2[indices]
    #batch_test_y = test_y[indices]
    
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
    

    
    with torch.no_grad( ) :
        embedding, output_prob,features = model(batch_test_x1.cuda(), batch_test_x2.cuda())
        embedding, output_prob2,features = model(batch_test_x2.cuda(), batch_test_x1.cuda())
    
    sftmax = torch.softmax(output_prob, dim=-1).detach().cpu()    
    prob = list(sftmax.numpy()) #list(softmax.detach().cpu().numpy())

    softmax2 = torch.softmax(output_prob2, dim=-1).detach().cpu() 
    prob2 = list(softmax2.numpy())

    #print(prob)
    #print('After Avg------------------')



    prob.extend(prob2)
    #print(np.array(prob).shape)
    prob = [np.average(prob, axis = 0)]
    prob = np.array(prob)

    
    pred = np.argmax(prob, axis=1)


    probabilities.append(prob[0][1])
    predictions.append(pred)

    batch_test_x1.detach()
    batch_test_x2.detach()    
    
    
print("PERFORMANCE ON TEST SET")    
compute_metrics(probabilities, predictions, test_y)



    
