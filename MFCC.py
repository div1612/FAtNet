import sys
import os

import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
from tqdm import tqdm

#For 3 second MFCC'

three_second_dev_audio_path = {}

three_sec_dev_wav_folder = "/home/divyas/Workspace/vox2/ThreeSecond/chunks/dev"
three_sec_dev_mfcc_folder = "/home/divyas/Workspace/vox2/ThreeSecond/MFCC/dev/"


for dirpath, dir, files in os.walk(three_sec_dev_wav_folder):
    
    for file in files:
        speaker = file.split('____')[0]
        
        if speaker not in three_second_dev_audio_path:
            three_second_dev_audio_path[speaker] = []
            
        path = dirpath+"/"+file
        three_second_dev_audio_path[speaker].append(path)
        
        
counter = 0 

for speaker in tqdm(three_second_dev_audio_path):
    
    if counter%1000==0:
        print(counter)
    counter = counter+1
    
    for wav_path in tqdm(three_second_dev_audio_path[speaker], leave=False):
        signal_values, sample_rate = librosa.load(wav_path, sr = None)
        mfcc_feat = librosa.feature.mfcc(y=signal_values, sr=16000, S=None, n_mfcc=80) # Earlier
        mfcc_feat = librosa.util.normalize(mfcc_feat)
        wav_name = wav_path.split('/')[8]
        dest_path = three_sec_dev_mfcc_folder+wav_name
        dest_path = dest_path.split('.')[0]
        dest_path = dest_path + ".npy"
        np.save(dest_path, mfcc_feat)
