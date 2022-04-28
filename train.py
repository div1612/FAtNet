from dataLoader import DataLoader
import utils
import models
import time
from tqdm import tqdm
import numpy as  np
from csv import writer


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


# Reference : https://thispointer.com/python-how-to-append-a-new-row-to-an-existing-csv-file/
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

        
        
        
# Load data
print("Loading Data ... ")
data = DataLoader() 
training_examples = data.training_examples
validation_examples = data.validation_examples
path_train_x1, path_train_x2, train_y =  data.get_path_train()
path_val_x1, path_val_x2, val_y = data.get_path_val() 
path_train_x1 = np.array(path_train_x1)
path_train_x2 = np.array(path_train_x2)
path_val_x1 = np.array(path_val_x1)
path_val_x2 = np.array(path_val_x2)
train_y = np.array(train_y)
val_y = np.array(val_y)



# Batch Size
batch_size = 128
print("Batch Size = {}".format(batch_size))

# Model name
model_name = "StAtNet"
    
print("Model Name = {}".format(model_name))


# Number of epochs to be executed
n_epochs = 350


# Define the model
#model = models.FAtNet()
#model = torch.load("/home/divyas/Workspace/AT/Vox2Code/Code/Files/Results/FAtNet_vox2/Models/26__1613044521.3617792.pt")
model = models.FAtNet2()
model = torch.load("/home/divyas/Workspace/AT/Vox2Code/Code/Files/Results/FAtNet2_vox2/Models/1__1615030009.6493034.pt")

# Define the criterion
softmaxloss = nn.CrossEntropyLoss()
centerloss = CenterLoss(2, 1024)
weight = 1#0.01

# optimzer4nn
optimizer4nn = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.00001) #lr=0.005
scheduler = lr_scheduler.StepLR(optimizer4nn, 10, gamma=0.5) #earlier step was 10

# optimzer4center
optimzer4center = optim.RMSprop(centerloss.parameters(),  lr=0.0005 ) # lr=0.2
scheduler_center = lr_scheduler.StepLR(optimzer4center, 10, gamma = 0.3) # earlier step was 10

# Use GPU
if torch.cuda.is_available():
    print("USING GPU")
    model = model.cuda()
    softmaxloss = softmaxloss.cuda()
    centerloss = centerloss.cuda()
    
    
    
# Loop for epochs
for epoch in (range(2,n_epochs+1)):
    
    torch.cuda.empty_cache()
    
    print("==============================================")
    print("Epoch {}".format(epoch))
    
    model.train()
    

    #scheduler.step()
        

    #scheduler_center.step()
    
    nn_lr = 0
    for param_group in  optimizer4nn.param_groups:
        print("Learning Rate = {}".format(param_group['lr']))
        nn_lr = param_group['lr']
    
    center_lr = 0
    for param_group in optimzer4center.param_groups:
        print("Learning Rate = {}".format(param_group['lr']))
        center_lr = param_group['lr']
        
        
    since = time.time()
    
    permutation = torch.randperm(len(training_examples))    
    
    training_accuracy = []
    training_true_positives = []
    training_true_negatives = []
    training_false_positives = []
    training_false_negatives = []
    training_softmax_loss = []
    training_center_loss = []
    training_total_loss = []
    
    total_loss = 0
    correct = 0
    total = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    
    # For every batch in training set
    for i in tqdm(range(0,len(training_examples), batch_size)):

        if i%10000==0:
            print(i)

            
        indices = permutation[i:i+batch_size]
        indices = indices.tolist()
        
        batch_path_x1  = path_train_x1[indices] 
        batch_path_x2 = path_train_x2[indices]
        batch_train_y = train_y[indices]
        
        batch_train_x1 = []
        for path in (batch_path_x1):
            mfcc1 = np.load(path)
            mfcc1 = mfcc1.astype(np.float32)
            mfcc1 = mfcc1.T
            batch_train_x1.append(mfcc1)
            
        batch_train_x1 = np.array(batch_train_x1)
        batch_train_x1 = torch.from_numpy(batch_train_x1)
        
        batch_train_x2 = []
        for path in (batch_path_x2):
            mfcc2 = np.load(path)
            mfcc2 = mfcc2.astype(np.float32)
            mfcc2 = mfcc2.T
            #mfcc2 = torch.from_numpy(mfcc2)
            batch_train_x2.append(mfcc2)
        
        batch_train_x2 = np.array(batch_train_x2)
        batch_train_x2 = torch.from_numpy(batch_train_x2)
        
        batch_train_y = np.array(batch_train_y)
        batch_train_y = batch_train_y.astype(int)
        batch_train_y = torch.from_numpy(batch_train_y)
        
        batch_train_x1 = Variable(batch_train_x1)
        batch_train_x2 = Variable(batch_train_x2)
        batch_train_y = Variable(batch_train_y)
        
        if torch.cuda.is_available():
            batch_train_x1 = batch_train_x1.cuda()
            batch_train_x2 = batch_train_x2.cuda()
            batch_train_y = batch_train_y.cuda()    
        
        optimizer4nn.zero_grad()
        optimzer4center.zero_grad()
        
        embedding, output_prob,  discriminative_features1,discriminative_features2 = model(batch_train_x1, batch_train_x2)
 
        loss1 = softmaxloss(output_prob, batch_train_y)
        loss2 = weight * centerloss(batch_train_y,embedding)
        loss = loss1 + loss2
        
        loss.backward()
        
        optimizer4nn.step()
        optimzer4center.step()
        
        training_softmax_loss.append(loss1.item())
        training_center_loss.append(loss2.item())
        training_total_loss.append(loss.item())
        
        softmax = torch.exp(output_prob.detach()).cpu()
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis=1)
        
        correct += (predictions == batch_train_y.cpu().detach().numpy()).sum().item()
        total += len(batch_train_y)
        
        tp = tp + sum([1 if (t == 1 and p == 1) else 0 for t, p in zip(predictions, batch_train_y.cpu().detach().numpy())])
        tn = tn + sum([1 if (t == 0 and p == 0) else 0 for t, p in zip(predictions, batch_train_y.cpu().detach().numpy())])
        fp = fp + sum([1 if (t == 0 and p == 1) else 0 for t, p in zip(predictions, batch_train_y.cpu().detach().numpy())])
        fn = fn + sum([1 if (t == 1 and p == 0) else 0 for t, p in zip(predictions, batch_train_y.cpu().detach().numpy())])

        
    
    
    training_softmax_loss = np.average(training_softmax_loss)
    training_center_loss = np.average(training_center_loss)
    training_total_loss = np.average(training_total_loss)
    training_accuracy.append(float(correct/total))
    training_true_positives.append(tp)
    training_true_negatives.append(tn)
    training_false_positives.append(fp)
    training_false_negatives.append(fn)
    
    time_elapsed = time.time() - since
    
    print('Epoch Time : {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  
    print('----------------------------------------------')
    print("Training Results:")
    print("Accuracy : {}".format(float(correct/total)))
    print("TP : {}".format(tp))
    print("TN : {}".format(tn))
    print("FP : {}".format(fp))
    print("FN : {}".format(fn))
    print("Softmax Loss : {}".format(training_softmax_loss))
    print("Center Loss : {}".format(training_center_loss))
    print("Total Loss : {}".format(training_total_loss))
    
    lst = [str(epoch),  ""+str(float(correct/total)),  str(tp), str(tn) , str(fp) , str(fn) , str(training_softmax_loss), str(training_center_loss), str(training_total_loss), str(nn_lr), str(center_lr) ]
    
    append_list_as_row('Files/Results/FAtNet2_vox2/TrainingStatisticAttention.csv', lst)
    
    
    
    
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    print('Validating ...')
    since = time.time()
    
    permutation = torch.randperm(len(validation_examples))
    validation_accuracy = []
    validation_softmax_loss = []
    validation_center_loss = []
    validation_total_loss = []
    validation_true_positives = []
    validation_true_negatives = []
    validation_false_positives = []
    validation_false_negatives = []
    
    total_loss = 0
    correct = 0
    total = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for i in tqdm(range(0,len(validation_examples), batch_size)):

        indices = permutation[i:i+batch_size]
        indices = indices.tolist()
        batch_path_x1  = path_val_x1[indices] 
        batch_path_x2 = path_val_x2[indices]
        batch_val_y = val_y[indices]
        
        batch_val_x1 = []
        for path in (batch_path_x1):    
            mfcc1 =  np.load(path)
            mfcc1 = mfcc1.astype(np.float32)
            mfcc1 = mfcc1.T
            #mfcc1 = torch.from_numpy(mfcc1)
            batch_val_x1.append(mfcc1)
         
        batch_val_x1 = np.array(batch_val_x1)
        batch_val_x1 = torch.from_numpy(batch_val_x1)
        #batch_val_x1 = torch.stack([j.squeeze(0) for j  in batch_val_x1])    
            
        
        batch_val_x2 = []
        for path in (batch_path_x2):
            mfcc2 = np.load(path)
            mfcc2 = mfcc2.astype(np.float32)
            mfcc2 = mfcc2.T
            #mfcc2 = torch.from_numpy(mfcc2)
            batch_val_x2.append(mfcc2)
            
        #batch_val_x2 = torch.stack([j.squeeze(0) for j in batch_val_x2])
        batch_val_x2 = np.array(batch_val_x2)
        batch_val_x2 = torch.from_numpy(batch_val_x2)
        
        batch_val_y = np.array(batch_val_y)
        batch_val_y = batch_val_y.astype(int)
        batch_val_y = torch.from_numpy(batch_val_y)
        
        batch_val_x1 = Variable(batch_val_x1)
        batch_val_x2 = Variable(batch_val_x2)
        batch_val_y = Variable(batch_val_y)
        
        if torch.cuda.is_available():
            batch_val_x1 = batch_val_x1.cuda()
            batch_val_x2 = batch_val_x2.cuda()
            batch_val_y = batch_val_y.cuda()
            
        with torch.no_grad( ) :
            embedding, output_prob,  discriminative_features1,discriminative_features2 = model(batch_val_x1, batch_val_x2)
            loss1 = softmaxloss(output_prob, batch_val_y)
            loss2 = weight * centerloss(batch_val_y,embedding)
            loss = loss1 + loss2
            
            validation_softmax_loss.append(loss1.item())
            validation_center_loss.append(loss2.item())
            validation_total_loss.append(loss.item())
            
            softmax = torch.exp(output_prob.detach()).cpu()
            prob = list(softmax.numpy())
            predictions = np.argmax(prob, axis=1)
            correct += (predictions == batch_val_y.cpu().detach().numpy()).sum().item()
            total += len(batch_val_y)

            tp = tp + sum([1 if (t == 1 and p == 1) else 0 for t, p in zip(predictions, batch_val_y.cpu().detach().numpy())])
            tn = tn + sum([1 if (t == 0 and p == 0) else 0 for t, p in zip(predictions, batch_val_y.cpu().detach().numpy())])
            fn = fn + sum([1 if (t == 1 and p == 0) else 0 for t, p in zip(predictions, batch_val_y.cpu().detach().numpy())])
    
    
    
    validation_softmax_loss = np.average(validation_softmax_loss)
    validation_center_loss = np.average(validation_center_loss)
    validation_total_loss = np.average(validation_total_loss)
    validation_accuracy.append(float(correct/total))
    validation_true_positives.append(tp)
    validation_true_negatives.append(tn)
    validation_false_positives.append(fp)
    validation_false_negatives.append(fn)
    
    time_elapsed = time.time() - since
    
    print('Epoch Time : {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  
    print("VALIDATION RESULTS: ")
    print("Accuracy : {}".format(float(correct/total)))
    print("TP : {}".format(tp))
    print("TN : {}".format(tn))
    print("FP : {}".format(fp))
    print("FN : {}".format(fn))
    print("Softmax Loss : {}".format(validation_softmax_loss))
    print("Center Loss : {}".format(validation_center_loss))
    print("Total Loss : {}".format(validation_total_loss))
    
    lst = [str(epoch),  ""+str(float(correct/total)),  str(tp), str(tn) , str(fp) , str(fn) , str(validation_softmax_loss), str(validation_center_loss), str(validation_total_loss) ]
    
    append_list_as_row('Files/Results/FAtNet2_vox2/ValidationStatisticAttention.csv', lst)
    
    print("==============================================")
    
    torch.save(model,"Files/Results/FAtNet2_vox2/Models/"+str(epoch)+"__"+str(time.time())+".pt")
