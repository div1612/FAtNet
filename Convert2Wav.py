# Objective: Convert the VoxCeleb speech recordings from aac to wav format

import sys
import numpy as np
import scipy.io.wavfile as wavfile
import audioread
import os
import soundfile as sf
from pydub import AudioSegment
from pydub.utils import make_chunks
#import librosa
#import librosa.display
import matplotlib.pyplot as plt
from numpy import random
#import cv2
from tqdm import tqdm
from csv import writer

# Reference : https://thispointer.com/python-how-to-append-a-new-row-to-an-existing-csv-file/
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


# Set the location of the source folder where the aac files are stored
folder_loc = "/home/divyas/Workspace/vox2/dev/aac" 

# Set the location of the target folder where the converted wav files are to be saved
target_loc = "/home/divyas/Workspace/vox2/dev_wav"



speaker_id = []

for dirpath, dir, files in os.walk(folder_loc):
    
    speaker_id = dir
    
    break
    

c=0
    
for speaker in tqdm(speaker_id):
    c=c+1
    
    lst = [str(c)]
    append_list_as_row('Progress.csv', lst)

    
    path = folder_loc+"/"+speaker
    folder = []
    
    for dirpath, dir, files in os.walk(path):
        folder = dir
        break
    
    
    for folder_name in tqdm(folder):
        file_loc = path+"/"+folder_name
        audio_files = []
        
        for dirpath, dir, files in os.walk(file_loc): 
            audio_files = files
            break
        
        for audio in tqdm(audio_files):                       
            conversion_name = audio.split('.')[0]
            os.system('ffmpeg -y -i '+file_loc+"/"+audio+' -sample_rate 16000 '+target_loc+"/"+speaker+"____"+folder_name+"____"+conversion_name+".wav" )
      

            
            
print(c)
