 # Testing StAtNet on VoxCeleb 1-test set using S4 strategy

from DataLoader import DataLoader
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

import librosa
import matplotlib.pyplot as plt
import librosa.display
import math
from tqdm.auto import tqdm
from scipy.io import wavfile
import os
import torch.nn as nn
import soundfile as sf


def cos_sim(a,b):
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))



# Load data
print("Loading Data ... ")
data = DataLoader() 
test_x1, test_x2, test_y = data.multilingual_test_S3()#data.aishell1_test_S3() #data.multilingual_test_S3()#data.voxceleb1_test_S3() 
#test_x1 = np.array(test_x1)
#test_x2 = np.array(test_x2)
test_y = np.array(test_y)
print("Voxforge")

# Batch Size
batch_size = 1
print("Batch Size = {}".format(batch_size))


# Load model
model = torch.load("/home/divyas/Workspace/AT/Vox2Code/Code/Files/Results/FAtNet2_vox2/Models/32__1615422992.5739744.pt")#60__1612722201.6210203_teacher.pt")#38__1613198443.9425883.pt")
model = model.cuda()

#https://github.com/zengchang94622/Speaker_Verification_Tencent/blob/master/inception_with_centloss.py
def eer(y_true, y_pred):
    fpr, tpr, thres = roc_curve(y_true, y_pred)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0, 1)
    thres = interp1d(fpr, thres)(eer)
    return eer, thres



def compute_metrics(output_prob, target):
    
    genuine_match_scores = []
    imposter_match_scores = []
    

    
    
    for i in tqdm(range(len(target))):
        
       
        if target[i]==1:
            genuine_match_scores.append(output_prob[i])
        elif target[i]==0:
            imposter_match_scores.append(output_prob[i])
            
            
    eer_value, threshold = eer(target, output_prob)
    

    print("Equal Error Rate (EER) : {}".format(eer_value))
    print("EER Threshold : {}".format(threshold))
    
    '''
    stats = get_eer_stats(genuine_match_scores, imposter_match_scores)
    generate_eer_report([stats], ['Testing'],'/home/divyas/Workspace/AT/Vox2Code/Code/Res/Speech_FAtNet2Vox2Train_VoxCeleb1_Test_S4/S4_FAtNet2_epoch32_Vox_eer_report.csv')
    export_error_rates(stats.fmr, stats.fnmr, '/home/divyas/Workspace/AT/Vox2Code/Code/Res/Speech_FAtNet2Vox2Train_VoxCeleb1_Test_S4/S4_FAtNet_epoch32Vox_Test_DET.csv')
    plot_eer_stats([stats], ['Testing'])
    
    #dict = {'genuine_match_scores': genuine_match_scores, 'imposter_match_scores': imposter_match_scores}
    #df = pd.DataFrame(dict) 
    #df.to_csv('/home/divyas/Workspace/AT/Vox2Code/Code/Res/Speech_FAtNetVox2Train_Voxforge_Test_S3/S3_FAtNet_epoch38_Vox_Test.csv')  
    
  
    
    with open('/home/divyas/Workspace/AT/Vox2Code/Code/Res/Speech_FAtNet2Vox2Train_VoxCeleb1_Test_S4/S4_FAtNet_epoch32_Test_genuine_match_scores.txt', mode='wt') as gscore:
        gscore.write('\n'.join(str(line) for line in genuine_match_scores))

    with open('/home/divyas/Workspace/AT/Vox2Code/Code/Res/Speech_FAtNet2Vox2Train_VoxCeleb1_Test_S4/S4_FAtNet_epoch32_Test_imposter_match_scores.txt', mode='wt') as iscore:
        iscore.write('\n'.join(str(line) for line in imposter_match_scores))
    
    '''
    

probabilities = []
predictions = []

model.eval()
lst = [i for i in range(0, len(test_y))]
print(len(test_y))

for i in tqdm((range(0,len(test_y), batch_size))): 

    indices = lst[i:i+batch_size]
    #print(indices)
    batch_path_x1  = np.array(test_x1[i])
    batch_path_x2 = np.array(test_x2[i])
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

    #print("batch_test_x1 :",batch_test_x1.shape)
    #print("batch_test_x2 :",batch_test_x2.shape)

    with torch.no_grad( ) :
        embedding, output_prob,audio1,audio2 = model(batch_test_x1.cuda(), batch_test_x2.cuda())
        embedding, output_prob,audio3,audio4 = model(batch_test_x2.cuda(), batch_test_x1.cuda())
        audio1 = (audio1+audio4)/2
        audio2 = (audio2+audio3)/2
        #print("audio1:",audio1.shape)
        #print("audio2:",audio2.shape)

        audio1 = audio1.mean(dim=1)
        audio2 = audio2.mean(dim=1)
        
        #print("audio1:",audio1.shape)
        #print("audio2:",audio2.shape)
        fatprob = []
        
        for j in range(audio1.shape[0]):
            fatprob.append(cos_sim(audio1[j].squeeze().detach().cpu().numpy(), audio2[j].squeeze().detach().cpu().numpy()))
    
    fatprob = np.array(fatprob)
    fatprob = np.mean(fatprob)
    #print("fatprob:",fatprob)
    probabilities.append(fatprob)
   

    batch_test_x1.detach()
    batch_test_x2.detach()    
  



    
    
    
print("PERFORMANCE ON TEST SET")    
compute_metrics(probabilities, test_y)



    
