# Import relevant libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import itertools


class DataLoader():
    
    
    
    def __init__(self):
        
        self.training_examples = pd.read_csv('Files/Vox2three_sec_training_examples.txt',header=None)#'Files/vox2Spectrogram_training_examples.txt', three_sec_training_examples.txt for VGG on VoxCeleb-1 dev, Vox2three_sec_training_examples.txt', header = None)#three_sec_training_examples.txt', header = None),  for medium VAD use Files/three_sec_training_examplesHighThresholdVAD.txt, Baselines/three_sec_training_examples.txt
        print("Number of training examples = {}".format(len(self.training_examples)))   
        self.training_examples = self.training_examples[0].to_list()
        
        self.validation_examples = pd.read_csv('Files/Vox2three_sec_validation_examples.txt',header=None)#'Files/vox2Spectrogram_validation_examples.txt', Files/Vox2three_sec_validation_examples.txt', header = None) #three_sec_validation_examples.txt', header = None), three_sec_validation_examplesHighThresholdVAD.txt, Baselines/three_sec_validation_examples.txt
        print("Number of validation examples = {}".format(len(self.validation_examples)))
        self.validation_examples = self.validation_examples[0].to_list()
        
        
        
            
    def get_path_train(self):   
        
        train_x1 = []
        train_x2 = []
        train_y = []
        
        for i in tqdm(range(len(self.training_examples))):
            example = self.training_examples[i]
            parts = example.split(' ')
            y = parts[0]
            x1 = parts[1]
            x2 = parts[2]
            train_x1.append(x1.strip())
            train_x2.append(x2.strip())
            train_y.append(int(y)) 
            
        return train_x1, train_x2, train_y
    
    
    
       
        
    def get_path_val(self):       
        
        val_x1 = []
        val_x2 = []
        val_y = []
        
        for i in tqdm(range(len(self.validation_examples))):
            example = self.validation_examples[i]
            parts = example.split(' ')
            y = parts[0]
            x1 = parts[1]
            x2 = parts[2]
            val_x1.append(x1.strip())
            val_x2.append(x2.strip())
            val_y.append(int(y))      
            
        return val_x1, val_x2, val_y
    
       
    
    def voxceleb1_test_S0(self):
        
        test_examples = pd.read_csv('/home/divyas/Workspace/AT/Vox2Code/Files/testing/voxceleb1/voxceleb1.txt', header = None)
        test_examples = test_examples[0].to_list()
        
        test_x1 = []
        test_x2 = []
        test_y = []
        
        for i in tqdm(range(len(test_examples))):
            
            example = test_examples[i]
            parts = example.split(' ')
            y = parts[0]
            
            x1 = parts[1]
            x1 = x1.split('.')[0]
            x1 = x1.split('/')
            x1 = '____'.join(x1)
            x1 = 'example'+'____'+str(i)+'____'+x1
            x1 = '/home/divyas/Workspace/AT/Vox2Code/Files/testing/voxceleb1/MFCC_S0/'+x1+'.npy'
            
            
            x2 = parts[2]
            x2 = x2.split('.')[0]
            x2 = x2.split('/')
            x2 = '____'.join(x2)
            x2 = 'example'+'____'+str(i)+'____'+x2
            x2 = '/home/divyas/Workspace/AT/Vox2Code/Files/testing/voxceleb1/MFCC_S0/'+x2+'.npy'
            
            test_x1.append(x1.strip())
            test_x2.append(x2.strip())
            test_y.append(int(y))
            
        return test_x1, test_x2, test_y
            
        

    
    
    
    
    
      
    def voxceleb1_test_S3(self):
        
        loc = "/home/divyas/Workspace/AT/Vox2Code/Files/testing/voxceleb1/MFCC_30/"

        for dirpath, dir, files in os.walk(loc):    
            break
    
        mfccs = files
        
        
        test_examples = pd.read_csv('/home/divyas/Workspace/AT/Vox2Code/Files/testing/voxceleb1/voxceleb1.txt', header = None)
        test_examples = test_examples[0].to_list()
        
        
        test_x1 = {}
        test_x2 = {}
        test_y = []
        
        for i in tqdm(range(len(test_examples))):
            

            example = test_examples[i]
            parts = example.split(' ')
            y = parts[0]
            
            x1 = parts[1]
            x1 = x1.split('.')[0]
            x1 = x1.split('/')
            x1 = '____'.join(x1)
            x1 = 'example'+'____'+str(i)+'____'+x1
            x1 = [(loc+file).strip() for file in mfccs if file.startswith(x1)]
            
            
            
            x2 = parts[2]
            x2 = x2.split('.')[0]
            x2 = x2.split('/')
            x2 = '____'.join(x2)
            x2 = 'example'+'____'+str(i)+'____'+x2
            x2 = [(loc+file).strip() for file in mfccs if file.startswith(x2)]
            
            combinations = list(itertools.product(x1,x2))
            x1 = [combinations[i][0] for i in range(len(combinations))]
            x2 = [combinations[i][1] for i in range(len(combinations))]
            
            
            test_x1[i] = x1
            test_x2[i] = x2
            test_y.append(int(y))
            
     
        return test_x1, test_x2, test_y
    
    
    
    
    
    
    def multi_test_vgg(self):
        
        #test_examples = pd.read_csv('testing/Speech/Voxforge.txt', header = None)
        test_examples = pd.read_csv('testing/Speech/Turkish.txt', header = None)
        #test_examples = pd.read_csv('testing/Speech/Hindi.txt', header = None)
        #test_examples = pd.read_csv('testing/Speech/Aishell1.txt', header = None)
        #test_examples = pd.read_csv('testing/Speech/LibriSpeech.txt', header = None)
        #print("Number of test examples = {}".format(len(test_examples)))
        test_examples = test_examples[0].to_list()
        
        test_x1 = []
        test_x2 = []
        test_y = []
        
        for i in tqdm(range(len(test_examples))):
            example = test_examples[i]
            parts = example.split(' ')
            y = parts[2]
            x1 = parts[0]
            x2 = parts[1]
            test_x1.append(x1.strip())
            test_x2.append(x2.strip())
            test_y.append(int(y))      
            
        return test_x1, test_x2, test_y
    
    
    
    
    
    
    
    def multilingual_test_S0(self):
        test_examples = pd.read_csv('testing/Speech/Voxforge.txt', header = None)
        test_examples = test_examples[0].to_list()
        
        test_x1 = []
        test_x2 = []
        test_y = []
        
        for i in tqdm(range(len(test_examples))):
            
            example = test_examples[i]
            parts = example.split(' ')
            y = parts[2]
            
            
            x1 = parts[0].split('.')[0]
            
            
            x1 = x1.split('/')
            x1 = "testing/MFCC_S0/"+x1[7]+"/"+x1[8]+"____"+x1[9]+"____"+x1[10]+".npy"
            '''
            if "Hindi" not in x1:
                x1 = "testing/MFCC_S0/"+x1[7]+"/"+x1[8]+"____"+x1[9]+"____"+x1[10]+".npy"
            else:
                x1 = "testing/MFCC_S0/Hindi/"+x1[len(x1)-1]+".npy"
            '''
            

            
            x2 = parts[1].split('.')[0]
            x2 = x2.split('/')
            x2 = "testing/MFCC_S0/"+x2[7]+"/"+x2[8]+"____"+x2[9]+"____"+x2[10]+".npy"
            '''
            if "Hindi" not in x2:
                x2 = "testing/MFCC_S0/"+x2[7]+"/"+x2[8]+"____"+x2[9]+"____"+x2[10]+".npy"
            else:
                x2 = "testing/MFCC_S0/Hindi/"+x2[len(x2)-1]+".npy"
            '''
            
            test_x1.append(x1.strip())
            test_x2.append(x2.strip())
            test_y.append(int(y))
            
        return test_x1, test_x2, test_y
            
 
    
    def multilingual_test_S3(self):
        loc = "/home/divyas/Workspace/AT/Vox2Code/Code/testing/MFCC_30/"

        
        test_examples = pd.read_csv('testing/Speech/Voxforge.txt', header = None)
        test_examples = test_examples[0].to_list()
        
        test_x1 = {}
        test_x2 = {}
        test_y = []
        
        for i in tqdm(range(len(test_examples))):
            
            example = test_examples[i]
            parts = example.split(' ')
            y = parts[2]
            
            
            x1 = parts[0].split('.')[0]
            
            
            x1 = x1.split('/')
            lan = x1[7]
            x1 = x1[8]+"____"+x1[9]+"____"+x1[10]
            #print('x1',x1)
            
            #print('lan = ',lan)
            loc_lan = loc+lan+"/"
            #print('loc_lan :',loc_lan)
            
            
            #print(loc_lan)
            for dirpath, dir, files in os.walk(loc_lan): 
                #print(files)
                break
    
            mfccs = files
            #print("mfccs : ",mfccs[0])
            x1 = [(loc_lan+file).strip() for file in mfccs if file.startswith(x1)]
            '''
            if "Hindi" not in x1:
                x1 = "testing/MFCC_S0/"+x1[7]+"/"+x1[8]+"____"+x1[9]+"____"+x1[10]+".npy"
            else:
                x1 = "testing/MFCC_S0/Hindi/"+x1[len(x1)-1]+".npy"
            '''
            

            
            x2 = parts[1].split('.')[0]
            x2 = x2.split('/')
            x2 = x2[8]+"____"+x2[9]+"____"+x2[10]
            x2 = [(loc_lan+file).strip() for file in mfccs if file.startswith(x2)]
            
            
            combinations = list(itertools.product(x1,x2))
            x1 = [combinations[i][0] for i in range(len(combinations))]
            x2 = [combinations[i][1] for i in range(len(combinations))]

            '''
            if "Hindi" not in x2:
                x2 = "testing/MFCC_S0/"+x2[7]+"/"+x2[8]+"____"+x2[9]+"____"+x2[10]+".npy"
            else:
                x2 = "testing/MFCC_S0/Hindi/"+x2[len(x2)-1]+".npy"
            '''
            
            test_x1[i] = x1
            test_x2[i] = x2
            test_y.append(int(y))
            #print(test_x1[i])
            #print(test_x2[i])
            #print(test_y)
            
            
        return test_x1, test_x2, test_y
            
        
        
   
    
    def aishell_multilingual_test_S0(self):
        test_examples = pd.read_csv('testing/Speech/Aishell1.txt', header = None)
        test_examples = test_examples[0].to_list()
        
        test_x1 = []
        test_x2 = []
        test_y = []
        
        for i in tqdm(range(len(test_examples))):
            
            example = test_examples[i]
            parts = example.split(' ')
            y = parts[2]
            
            
            x1 = parts[0].split('.')[0]
            
            
            x1 = x1.split('/')
            x1 = "testing/Aishell1MFCC_S0/"+x1[len(x1)-2]+'____'+x1[len(x1)-1]+".npy"
            '''
            if "Hindi" not in x1:
                x1 = "testing/MFCC_S0/"+x1[7]+"/"+x1[8]+"____"+x1[9]+"____"+x1[10]+".npy"
            else:
                x1 = "testing/MFCC_S0/Hindi/"+x1[len(x1)-1]+".npy"
            '''
            

            
            x2 = parts[1].split('.')[0]
            x2 = x2.split('/')
            x2 = "testing/Aishell1MFCC_S0/"+x2[len(x2)-2]+'____'+x2[len(x2)-1]+".npy"
            '''
            if "Hindi" not in x2:
                x2 = "testing/MFCC_S0/"+x2[7]+"/"+x2[8]+"____"+x2[9]+"____"+x2[10]+".npy"
            else:
                x2 = "testing/MFCC_S0/Hindi/"+x2[len(x2)-1]+".npy"
            '''
            
            test_x1.append(x1.strip())
            test_x2.append(x2.strip())
            test_y.append(int(y))
            
        return test_x1, test_x2, test_y
            
   




    def aishell1_test_S3(self):
        loc = "/home/divyas/Workspace/AT/Vox2Code/Code/testing/Aishell1MFCC_30/"

        
        test_examples = pd.read_csv('testing/Speech/Aishell1.txt', header = None)
        test_examples = test_examples[0].to_list()
        
        test_x1 = {}
        test_x2 = {}
        test_y = []
        
        for i in tqdm(range(len(test_examples))):
            
            example = test_examples[i]
            parts = example.split(' ')
            y = parts[2]
            
            
            x1 = parts[0].split('.')[0]
            
            
            x1 = x1.split('/')
            #lan = x1[7]
            x1 = x1[len(x1)-2]+"____"+x1[len(x1)-1]
            #print('x1',x1)
            
            #print('lan = ',lan)
            loc_lan = loc+"/"
            #print('loc_lan :',loc_lan)
            
            
            #print(loc_lan)
            
            for dirpath, dir, files in os.walk(loc_lan): 
                #print(files)
                break
    
            mfccs = files
            #print("mfccs : ",mfccs[0])
            x1 = [(loc_lan+file).strip() for file in mfccs if file.startswith(x1)]
            '''
            if "Hindi" not in x1:
                x1 = "testing/MFCC_S0/"+x1[7]+"/"+x1[8]+"____"+x1[9]+"____"+x1[10]+".npy"
            else:
                x1 = "testing/MFCC_S0/Hindi/"+x1[len(x1)-1]+".npy"
            '''
            

            
            x2 = parts[1].split('.')[0]
            x2 = x2.split('/')
            x2 = x2[len(x2)-2]+"____"+x2[len(x2)-1]
            x2 = [(loc_lan+file).strip() for file in mfccs if file.startswith(x2)]
            
            
            combinations = list(itertools.product(x1,x2))
            x1 = [combinations[i][0] for i in range(len(combinations))]
            x2 = [combinations[i][1] for i in range(len(combinations))]

            '''
            if "Hindi" not in x2:
                x2 = "testing/MFCC_S0/"+x2[7]+"/"+x2[8]+"____"+x2[9]+"____"+x2[10]+".npy"
            else:
                x2 = "testing/MFCC_S0/Hindi/"+x2[len(x2)-1]+".npy"
            '''
            
            test_x1[i] = x1
            test_x2[i] = x2
            test_y.append(int(y))
            #print(test_x1[i])
            #print(test_x2[i])
            #print(test_y)

            
        return test_x1, test_x2, test_y
       

    
    
    
    
    def librispeech_test_S3(self):
        loc = "/home/divyas/Workspace/AT/Vox2Code/Code/testing/LibriSpeechMFCC_30/"

        
        test_examples = pd.read_csv('testing/Speech/LibriSpeech.txt', header = None)
        test_examples = test_examples[0].to_list()
        
        test_x1 = {}
        test_x2 = {}
        test_y = []
        
        for i in tqdm(range(len(test_examples))):
            
            example = test_examples[i]
            parts = example.split(' ')
            y = parts[2]
            
            
            x1 = parts[0].split('.')[0]
            
            
            x1 = x1.split('/')
            #lan = x1[7]
            x1 = x1[len(x1)-1]
            #print('x1',x1)
            
            #print('lan = ',lan)
            loc_lan = loc+"/"
            #print('loc_lan :',loc_lan)
            
            
            #print(loc_lan)
            
            for dirpath, dir, files in os.walk(loc_lan): 
                #print(files)
                break
    
            mfccs = files
            #print("mfccs : ",mfccs[0])
            x1 = [(loc_lan+file).strip() for file in mfccs if file.startswith(x1)]
            '''
            if "Hindi" not in x1:
                x1 = "testing/MFCC_S0/"+x1[7]+"/"+x1[8]+"____"+x1[9]+"____"+x1[10]+".npy"
            else:
                x1 = "testing/MFCC_S0/Hindi/"+x1[len(x1)-1]+".npy"
            '''
            

            
            x2 = parts[1].split('.')[0]
            x2 = x2.split('/')
            x2 = x2[len(x2)-1]
            x2 = [(loc_lan+file).strip() for file in mfccs if file.startswith(x2)]
            
            
            combinations = list(itertools.product(x1,x2))
            x1 = [combinations[i][0] for i in range(len(combinations))]
            x2 = [combinations[i][1] for i in range(len(combinations))]

            '''
            if "Hindi" not in x2:
                x2 = "testing/MFCC_S0/"+x2[7]+"/"+x2[8]+"____"+x2[9]+"____"+x2[10]+".npy"
            else:
                x2 = "testing/MFCC_S0/Hindi/"+x2[len(x2)-1]+".npy"
            '''
            
            test_x1[i] = x1
            test_x2[i] = x2
            test_y.append(int(y))
            #print(test_x1[i])
            #print(test_x2[i])
            #print(test_y)
           
            
        return test_x1, test_x2, test_y
    
    
    
    
    
    def multilingual_test_Aishell1S3(self):
        loc = "/home/divyas/Workspace/AT/Vox2Code/Code/testing/MFCC_30/"

        
        test_examples = pd.read_csv('testing/Speech/Aishell1.txt', header = None)
        test_examples = test_examples[0].to_list()
        
        test_x1 = {}
        test_x2 = {}
        test_y = []
        
        for i in tqdm(range(len(test_examples))):
            
            example = test_examples[i]
            parts = example.split(' ')
            y = parts[2]
            
            
            x1 = parts[0].split('.')[0]
            
            
            x1 = x1.split('/')
            lan = x1[7]
            x1 = x1[8]+"____"+x1[9]+"____"+x1[10]
            #print('x1',x1)
            
            #print('lan = ',lan)
            loc_lan = loc+lan+"/"
            #print('loc_lan :',loc_lan)
            
            
            #print(loc_lan)
            for dirpath, dir, files in os.walk(loc_lan): 
                #print(files)
                break
    
            mfccs = files
            #print("mfccs : ",mfccs[0])
            x1 = [(loc_lan+file).strip() for file in mfccs if file.startswith(x1)]
            '''
            if "Hindi" not in x1:
                x1 = "testing/MFCC_S0/"+x1[7]+"/"+x1[8]+"____"+x1[9]+"____"+x1[10]+".npy"
            else:
                x1 = "testing/MFCC_S0/Hindi/"+x1[len(x1)-1]+".npy"
            '''
            

            
            x2 = parts[1].split('.')[0]
            x2 = x2.split('/')
            x2 = x2[8]+"____"+x2[9]+"____"+x2[10]
            x2 = [(loc_lan+file).strip() for file in mfccs if file.startswith(x2)]
            
            
            combinations = list(itertools.product(x1,x2))
            x1 = [combinations[i][0] for i in range(len(combinations))]
            x2 = [combinations[i][1] for i in range(len(combinations))]

            '''
            if "Hindi" not in x2:
                x2 = "testing/MFCC_S0/"+x2[7]+"/"+x2[8]+"____"+x2[9]+"____"+x2[10]+".npy"
            else:
                x2 = "testing/MFCC_S0/Hindi/"+x2[len(x2)-1]+".npy"
            '''
            
            test_x1[i] = x1
            test_x2[i] = x2
            test_y.append(int(y))
            #print(test_x1[i])
            #print(test_x2[i])
            #print(test_y)
            break
            
        return test_x1, test_x2, test_y
            

        
        
        
    def language_test_S0(self):
        test_examples = pd.read_csv('/home/divyas/Workspace/AT/Vox2Code/Code/testing/Speech/Turkish.txt', header = None)
        test_examples = test_examples[0].to_list()

        test_x1 = []
        test_x2 = []
        test_y = []

        for i in tqdm(range(len(test_examples))):

            example = test_examples[i]
            parts = example.split(' ')
            y = parts[2]


            x1 = parts[0].split('.')[0]


            x1 = x1.split('/')
            x1 = "/home/divyas/Workspace/AT/Vox2Code/Code/testing/Exp3/TurkishMFCC_S0/"+x1[len(x1)-3]+"____"+x1[len(x1)-2]+"____"+x1[len(x1)-1]+".npy"

            '''
            if "Hindi" not in x1:
                x1 = "testing/MFCC_S0/"+x1[7]+"/"+x1[8]+"____"+x1[9]+"____"+x1[10]+".npy"
            else:
                x1 = "testing/MFCC_S0/Hindi/"+x1[len(x1)-1]+".npy"
            '''



            x2 = parts[1].split('.')[0]
            x2 = x2.split('/')
            x2 = "/home/divyas/Workspace/AT/Vox2Code/Code/testing/Exp3/TurkishMFCC_S0/"+x2[len(x2)-3]+"____"+x2[len(x2)-2]+"____"+x2[len(x2)-1]+".npy"
            '''
            if "Hindi" not in x2:
                x2 = "testing/MFCC_S0/"+x2[7]+"/"+x2[8]+"____"+x2[9]+"____"+x2[10]+".npy"
            else:
                x2 = "testing/MFCC_S0/Hindi/"+x2[len(x2)-1]+".npy"
            '''

            test_x1.append(x1.strip())
            test_x2.append(x2.strip())
            test_y.append(int(y))

        return test_x1, test_x2, test_y

    
    
    

      
