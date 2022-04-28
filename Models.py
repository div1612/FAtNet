import torch
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from torchvision import models
from torchvision import transforms
import torchvision.models as models
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm1d, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torch.nn as nn
import copy
import math 
import torch.nn.functional as F
import utils
#from utils import MultiHeadedAttention
from TDNN import TDNN
import numpy as np
from tqdm import tqdm
from collections import OrderedDict


# Reference : https://discuss.pytorch.org/t/attention-in-image-classification/80147/3
class Attention(Module):

    def __init__(self, indim = 74):
        
        super(Attention, self).__init__()

        self.q = nn.Conv1d(in_channels = indim , out_channels = indim , kernel_size= 1)
        self.k = nn.Conv1d(in_channels = indim , out_channels = indim , kernel_size= 1)
        self.v = nn.Conv1d(in_channels = indim , out_channels = indim , kernel_size= 1)
        
        
        
        return
    
    def forward(self, query, key, value):
        
        #print("================ ATTENTION =============")
        #print("query.shape : {}".format(query.shape))  # torch.Size([16, 743, 1024])
        
        
        query2 = self.q(query) 
        key2 = self.k(key) 
        value2 = self.v(value) 
        #print("query2.shape : {}".format(query2.shape)) # torch.Size([64, 743, 1024])
        #print("key2.shape : {}".format(key2.shape)) # torch.Size([64, 743, 1024])
        #print("value2.shape : {}".format(value2.shape)) # torch.Size([64, 743, 1024])
                         
      
        key2 = key2.reshape([key2.shape[0], key2.shape[2], key2.shape[1]]) 
        
        #print("After reshape, key2.shape : {}".format(key2.shape)) # torch.Size([16, 1024, 743])
        
        energy =  torch.bmm(query2,key2) # [b, 743, 1024] * [b, 1024, 743] = [b, 743, 743]
        #print("Energy : {}".format(energy.shape)) # torch.Size([64, 743, 743])
        
        p_attn  = F.softmax(input = energy , dim = -1 )
        
        #print("p_attn.shape : {}".format(p_attn.shape)) #  torch.Size([64, 743, 743])
        
        out = torch.bmm(p_attn, value2 ) #  [b, 743, 743] * [b,743, 1024] = [b,743,1024]       
                
        #print("out : {}".format(out.shape)) # torch.Size([16, 743, 1024])
      

        output = out+query
        #print("out+query : {}".format(output.shape)) # torch.Size([16, 743, 1024])
      
        return output
    
    
    
    
class Attention2(Module):

    def __init__(self, indim = 1024):
        
        super(Attention2, self).__init__()
        
        self.batchNorm_1 = nn.BatchNorm1d(num_features = 743)
        self.weights = nn.Linear(743,743)
        self.act = nn.Tanh()
        
        
    def forward(self, x):
        temp = x.clone()
        temp = temp.mean(dim=2)
        #print("temp.mean(dim=2) : {}".format(temp.shape)) # torch.Size([16, 743])
        temp = self.batchNorm_1(temp)
        #print("After batchnorm : {}".format(temp.shape)) # torch.Size([16, 743])
        wght = self.weights(temp)
        wght = self.act(wght)
        #print("After passing througn a linear layer for weights : {}".format(wght.shape)) # torch.Size([16, 743])
        #print(x.shape) # torch.Size([16, 743, 1024])
        #print(wght.unsqueeze(2).shape) # torch.Size([16, 743, 1])
        x = x+wght.unsqueeze(2)*x
        #print("Ouput.shape : {}".format(x.shape))  #  torch.Size([16, 743, 1024])      
        return x
        
        
        

class FAtNetv1(Module):
    
        
    def __init__(self):
        
        super(FAtNetv1, self).__init__()
        
        self.avg_pool1 = nn.AdaptiveAvgPool2d((94,80)) # changing 80 to 20
        self.avg_pool2 = nn.AdaptiveAvgPool2d((94,80)) #changing 80 to 20
        
        # Extract frame-level features
        self.tdnn1_layer1 = TDNN(input_dim=80, output_dim=512, context_size=7, dilation=1, batch_norm = True) 
        self.tdnn2_layer1 = TDNN(input_dim=80, output_dim=512, context_size=7, dilation=1, batch_norm = True) 
        
        self.tdnn1_layer2 = TDNN(input_dim=512,output_dim=512, context_size=5, dilation=2, batch_norm = True)
        self.tdnn2_layer2 = TDNN(input_dim=512,output_dim=512, context_size=5, dilation=2, batch_norm = True)
        
        self.tdnn1_layer3 = TDNN(input_dim=512,output_dim=512, context_size=3, dilation=3, batch_norm = True)
        self.tdnn2_layer3 = TDNN(input_dim=512,output_dim=512, context_size=3, dilation=3,batch_norm = True)
        
        self.tdnn1_layer4 = TDNN(input_dim=512,output_dim=1024, context_size=1, dilation=1,batch_norm = True)
        self.tdnn2_layer4 = TDNN(input_dim=512,output_dim=1024, context_size=1, dilation=1, batch_norm = True)
      
        
        self.batchNorm_1 = nn.BatchNorm1d(num_features = 2048)
        self.batchNorm_2 = nn.BatchNorm1d(num_features = 2048)
        self.batchNorm_3 = nn.BatchNorm1d(num_features = 2048)
       
        
       
        
        self.attention1 = Attention(indim = 74)
        #self.attention2 = Attention2(indim = 1024)
        
        self.fc3 = nn.Linear(2048,2048)
        self.fc4 = nn.Linear(2048, 2048)
        
        self.fc_classifier = nn.Linear(2048,2)
        
        
    def forward(self, mfcc1, mfcc2):
        
        #print("mfcc1.shape : {}".format(mfcc1.shape)) # [64, 94, 80]
        #print("mfcc2.shape : {}".format(mfcc2.shape)) # [64, 94, 80]
        
        avg1 = self.avg_pool1(mfcc1)
        avg2 = self.avg_pool2(mfcc2)
        
        #print("avg1.shape : {}".format(avg1.shape)) # torch.Size([16, 751, 80])
        #print("avg2.shape : {}".format(avg2.shape)) # torch.Size([16, 751, 80])
        
        # Extract frame-level features
        feature1_layer1 = self.tdnn1_layer1(avg1)
        feature2_layer1 = self.tdnn2_layer1(avg2)
        feature1_layer2 = self.tdnn1_layer2(feature1_layer1)
        feature2_layer2 = self.tdnn2_layer2(feature2_layer1)
        feature1_layer3 = self.tdnn1_layer3(feature1_layer2)
        feature2_layer3 = self.tdnn2_layer3(feature2_layer2)
        feature1_layer4 = self.tdnn1_layer4(feature1_layer3)
        feature2_layer4 = self.tdnn2_layer4(feature2_layer3)

        #print("feature1_layer4  : {}".format(feature1_layer4.shape)) # [64, 74, 1024]
        #print("feature2_layer4  : {}".format(feature2_layer4.shape)) # [64, 74, 1024]
        
        # Concatenate
        concatenate = torch.cat( (feature1_layer4, feature2_layer4), 2 )
        #print("concatenate  : {}".format(concatenate.shape)) # torch.Size([64, 74, 2048])
        concatenate = concatenate.permute(0,2,1)
        concatenate = self.batchNorm_1(concatenate)
        concatenate = concatenate.permute(0,2,1)
        #print("BatchNorm after concatenate  : {}".format(concatenate.shape)) # torch.Size([64, 74, 2048])
        
               
     
        #print("--------------------- ATTENTION BLOCK 1-------------------------")
        #print("Inside attention 1 block")
        attn = self.attention1(concatenate.clone(), concatenate.clone(), concatenate.clone())
        #print("Outside attention 1 block")
        #print("--------------------- OUT OF ATTENTION BLOCK -------------------------")
        attn = attn.permute(0,2,1)
        attn = self.batchNorm_2(attn)
        attn = attn.permute(0,2,1)
        #print("After batchNorm :{}".format(attn.shape)) # [64, 74, 2048]
        
        #print("--------------------- ATTENTION BLOCK 2-------------------------")
        #print("Inside attention 2 block")
        #attn = self.attention2(attn)
        #print("Outside attention 2 block") 
        attn = attn.mean(dim=1)
        #print("Mean at dim=1 : {}".format(attn.shape)) # torch.Size([16, 1024])
        #print("--------------------- OUT OF ATTENTION BLOCK -------------------------")
 
        embedding = self.fc3(attn)
        embedding = self.batchNorm_3(embedding)
        #print("embedding after fc3 : {}".format(embedding.shape))  # torch.Size([16, 1024])
        embedding = F.leaky_relu(embedding)
        
        
        embedding = self.fc4(embedding)
        embedding = F.normalize(embedding, dim=1,p=2)
        embedding = F.leaky_relu(embedding)
        #print("embedding after fc4 : {}".format(embedding.shape)) #torch.Size([16, 1024])
        
        # Classifier
        result = self.fc_classifier(embedding)
        #print("result : {}".format(result.shape)) # [64, 2]
        
        return embedding,  result, feature1_layer4, feature2_layer4

    

    
    
    
class FAtNet2(Module):
    
        
    def __init__(self):
        
        super(FAtNet2, self).__init__()
        
        self.avg_pool1 = nn.AdaptiveAvgPool2d((751,80)) # changing 80 to 20
        self.avg_pool2 = nn.AdaptiveAvgPool2d((751,80)) #changing 80 to 20
        
        # Extract frame-level features
        self.tdnn1_layer1 = TDNN(input_dim=80, output_dim=512, context_size=3, dilation=1,dropout_p=0.1) #changing 80 to 20
        self.tdnn2_layer1 = TDNN(input_dim=80, output_dim=512, context_size=3, dilation=1,dropout_p=0.1) #changing 80 to 20
        
        self.tdnn1_layer2 = TDNN(input_dim=512, context_size=5, dilation=1,dropout_p=0.1)
        self.tdnn2_layer2 = TDNN(input_dim=512, context_size=5, dilation=1,dropout_p=0.1)
        
        self.tdnn1_layer3 = TDNN(input_dim=512, context_size=3, dilation=1,dropout_p=0.1, batch_norm = True)
        self.tdnn2_layer3 = TDNN(input_dim=512, context_size=3, dilation=1,dropout_p=0.1, batch_norm = True)
        
        self.tdnn1_layer4 = TDNN(input_dim=512, context_size=1, dilation=1,dropout_p=0.1, batch_norm = True)
        self.tdnn2_layer4 = TDNN(input_dim=512, context_size=1, dilation=1,dropout_p=0.1, batch_norm = True)
  
        
        self.batchNorm_1 = nn.BatchNorm1d(num_features = 1024)
 
       
        self.fc3 = nn.Linear(1024,1024)
       
        
        self.attention1 = Attention(l = 86, d = 512, dv = 128, dout = 512, nv = 4)
        self.attention2 = Attention(l = 86, d = 512, dv = 128, dout = 512, nv = 4)
        
        self.fc_classifier = nn.Linear(1024,2)
        
        
    def forward(self, mfcc1, mfcc2):
        
        avg1 = self.avg_pool1(mfcc1)
        avg2 = self.avg_pool2(mfcc2)
        
        # Extract frame-level features
        feature1_layer1 = self.tdnn1_layer1(avg1)
        feature2_layer1 = self.tdnn2_layer1(avg2)
        feature1_layer2 = self.tdnn1_layer2(feature1_layer1)
        feature2_layer2 = self.tdnn2_layer2(feature2_layer1)
        feature1_layer3 = self.tdnn1_layer3(feature1_layer2)
        feature2_layer3 = self.tdnn2_layer3(feature2_layer2)
        feature1_layer4 = self.tdnn1_layer4(feature1_layer3)
        feature2_layer4 = self.tdnn2_layer4(feature2_layer3)
        #print("feature1_layer1  : {}".format(feature1_layer1.shape))
        #print("feature1_layer2  : {}".format(feature1_layer2.shape))
        #print("feature1_layer3  : {}".format(feature1_layer3.shape))
        #print("feature1_layer4  : {}".format(feature1_layer4.shape))
        #print("feature2_layer4  : {}".format(feature2_layer4.shape))
        attn1 = self.attention1(feature1_layer4)
        attn2 = self.attention2(feature2_layer4)
        #print("attn1 : {}".format(attn1.shape))
        #print("attn2 : {}".format(attn2.shape))
              
        # Concatenate
        concatenate = torch.cat( (attn1, attn2), 2 )
        concatenate = concatenate.permute(0,2,1)
        concatenate = self.batchNorm_1(concatenate)
        concatenate = concatenate.permute(0,2,1)
        #print("concatenate  : {}".format(concatenate.shape)) # torch.Size([128, 86, 1024])
               
        # Path 2: Statistic-Attention
        #print("part2 after fc2 : {}".format(part2.shape)) # torch.Size([128, 86, 512])
        #print("--------------------- ATTENTION BLOCK -------------------------")
        
        #print("--------------------- OUT OF ATTENTION BLOCK -------------------------")

        #print("attn : {}".format(attn.shape)) #  torch.Size([128, 86, 1024]) 
        embedding2 = self.fc3(concatenate)
        #print("embedding2 after fc3 : {}".format(embedding2.shape)) # torch.Size([128, 86, 512])
        embedding2 = F.normalize(embedding2, dim=1,p=2)
        embedding2 = F.leaky_relu(embedding2)
        embedding2 = embedding2.mean(dim=1)
        #print("embedding2 after mean : {}".format(embedding2.shape)) # torch.Size([128, 512])
        
        # Classifier
        result = self.fc_classifier(embedding2)
        #print("result : {}".format(result.shape)) # torch.Size([128, 2])
        
        return embedding2,  result, attn1,attn2

    
