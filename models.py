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



# Reference : https://medium.com/@moshnoi2000/all-you-need-is-attention-computer-vision-edition-dbe7538330a4
class Attention(Module):

    def __init__(self, l = 86, d = 512, dv = 64, dout = 512, nv = 8):
        
        super(Attention, self).__init__()
        
        self.l = l
        self.d = d
        self.dv = dv
        self.dout = dout
        self.nv = nv
        
        self.q = nn.Linear(d, dv*nv)
        self.k = nn.Linear(d, dv*nv)
        self.v = nn.Linear(d, dv*nv)
        
        #self.tanh_q = nn.Tanh()
        #self.tanh_k = nn.Tanh()
        #self.tanh_v = nn.Tanh()
        
        self.fc = nn.Linear(d, dout)
        
        #self.tanh_res = nn.Tanh()
        
        
        return
    
    def forward(self, query, key, value):
        query2 = self.q(query)
        key2 = self.k(key)
        value2 = self.v(value)        
        
        query2 = F.relu(query2)
        key2 = F.relu(key2)
        value2 = F.relu(value2)
        
           
        query2 = query2.reshape([query2.shape[0], self.l, self.nv, self.dv])
        key2 = key2.reshape([key2.shape[0], self.l, self.nv, self.dv])
        value2 = value2.reshape([value2.shape[0], self.l, self.nv, self.dv])

        
        dot_product = [ torch.bmm( query2[i],key2[i].permute(0,2,1) ) for i in tqdm(range(query2.shape[0]), leave = False)]
        dot_product = torch.stack(dot_product)
        
        scaled_dot_product = dot_product/np.sqrt(self.dv)
                
        p_attn  = F.softmax(input = scaled_dot_product , dim = -1 )        
        
        residual_dot_product = [ torch.bmm( p_attn[i], value2[i] ) for i in tqdm(range(p_attn.shape[0]), leave = False)]
        
        residual_dot_product = torch.stack(residual_dot_product)
              
        output = residual_dot_product.reshape(residual_dot_product.shape[0], self.l, self.d)
        
        output = output+query
        
        output = self.fc(output)
        
        output = F.relu(output)
                 
        return output
    
    
    
    

    
    
    

    
    
class FAtNet(Module):
    
        
    def __init__(self):
        
        super(FAtNet, self).__init__()
        
        self.avg_pool1 = nn.AdaptiveAvgPool2d((94,80)) 
        self.avg_pool2 = nn.AdaptiveAvgPool2d((94,80)) 
        
        # Extract frame-level features
        self.tdnn1_layer1 = TDNN(input_dim=80, output_dim=512, context_size=3, dilation=1,dropout_p=0.1) 
        self.tdnn2_layer1 = TDNN(input_dim=80, output_dim=512, context_size=3, dilation=1,dropout_p=0.1)
        
        self.tdnn1_layer2 = TDNN(input_dim=512, context_size=5, dilation=1,dropout_p=0.1)
        self.tdnn2_layer2 = TDNN(input_dim=512, context_size=5, dilation=1,dropout_p=0.1)
        
        self.tdnn1_layer3 = TDNN(input_dim=512, context_size=3, dilation=1,dropout_p=0.1, batch_norm = True)
        self.tdnn2_layer3 = TDNN(input_dim=512, context_size=3, dilation=1,dropout_p=0.1, batch_norm = True)
        
        self.tdnn1_layer4 = TDNN(input_dim=512, context_size=1, dilation=1,dropout_p=0.1, batch_norm = True)
        self.tdnn2_layer4 = TDNN(input_dim=512, context_size=1, dilation=1,dropout_p=0.1, batch_norm = True)
  
        
        self.batchNorm_1 = nn.BatchNorm1d(num_features = 1024)
        self.batchNorm_4 = nn.BatchNorm1d(num_features = 1024)
       
        self.fc3 = nn.Linear(1024,1024)
       
        
        self.attention = Attention(l = 86, d = 1024, dv = 128, dout = 1024, nv = 8)
        
        self.fc_classifier = nn.Linear(1024,2)
        
        
    def forward(self, mfcc1, mfcc2):
        
        #print("mfcc1.shape : {}".format(mfcc1.shape))
        #print("mfcc2.shape : {}".format(mfcc2.shape))
        
        avg1 = self.avg_pool1(mfcc1)
        avg2 = self.avg_pool2(mfcc2)
        
        #print("avg1.shape : {}".format(avg1.shape))
        #print("avg2.shape : {}".format(avg2.shape))
        
        # Extract frame-level features
        feature1_layer1 = self.tdnn1_layer1(avg1)
        feature2_layer1 = self.tdnn2_layer1(avg2)
        feature1_layer2 = self.tdnn1_layer2(feature1_layer1)
        feature2_layer2 = self.tdnn2_layer2(feature2_layer1)
        feature1_layer3 = self.tdnn1_layer3(feature1_layer2)
        feature2_layer3 = self.tdnn2_layer3(feature2_layer2)
        feature1_layer4 = self.tdnn1_layer4(feature1_layer3)
        feature2_layer4 = self.tdnn2_layer4(feature2_layer3)

        
        # Concatenate
        concatenate = torch.cat( (feature1_layer4, feature2_layer4), 2 )
        concatenate = concatenate.permute(0,2,1)
        concatenate = self.batchNorm_1(concatenate)
        concatenate = concatenate.permute(0,2,1)

               
        # Path 2: Attention
        attn = self.attention(concatenate, concatenate, concatenate)
        attn = attn.permute(0,2,1)
        attn = self.batchNorm_4(attn)
        attn = attn.permute(0,2,1)
        embedding2 = self.fc3(attn)
        embedding2 = F.normalize(embedding2, dim=1,p=2)
        embedding2 = F.leaky_relu(embedding2)
        embedding2 = embedding2.mean(dim=1)
        
        # Classifier
        result = self.fc_classifier(embedding2)
        
        return embedding2,  result, feature1_layer4

    
    
    
    
    
    
class FAtNet2(Module):
    
        
    def __init__(self):
        
        super(FAtNet2, self).__init__()
        
        self.avg_pool1 = nn.AdaptiveAvgPool2d((94,80))
        self.avg_pool2 = nn.AdaptiveAvgPool2d((94,80)) 
        
        # Extract frame-level features
        self.tdnn1_layer1 = TDNN(input_dim=80, output_dim=512, context_size=3, dilation=1,dropout_p=0.1) 
        self.tdnn2_layer1 = TDNN(input_dim=80, output_dim=512, context_size=3, dilation=1,dropout_p=0.1) 
        
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

        attn1 = self.attention1(feature1_layer4, feature1_layer4, feature1_layer4)
        attn2 = self.attention2(feature2_layer4, feature2_layer4, feature2_layer4)

              
        # Concatenate
        concatenate = torch.cat( (attn1, attn2), 2 )
        concatenate = concatenate.permute(0,2,1)
        concatenate = self.batchNorm_1(concatenate)
        concatenate = concatenate.permute(0,2,1)

               
        # Path 2: Attention
        embedding2 = self.fc3(concatenate)
       
        embedding2 = F.normalize(embedding2, dim=1,p=2)
        embedding2 = F.leaky_relu(embedding2)
        embedding2 = embedding2.mean(dim=1)
      
        
        # Classifier
        result = self.fc_classifier(embedding2)
 
        return embedding2,  result, attn1,attn2
