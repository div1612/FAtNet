import torch
import torch.nn as nn
from torch.autograd.function import Function
import copy
import math 
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from torchvision import models
from torchvision import transforms
import torchvision.models as models
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm1d, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type(torch.cuda.FloatTensor)


# Reference : https://github.com/zengchang94622/Speaker_Verification_Tencent/blob/master/inception_with_centloss.py
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim)) 
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim # embedding dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1) # tensor([batch_size])
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        return loss

    
    
# Reference : https://github.com/zengchang94622/Speaker_Verification_Tencent/blob/master/inception_with_centloss.py    
class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size) 
        centers_batch = centers.index_select(0, label.long()) 
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature 
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0)) 
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff) 
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None
    
    
    
    

