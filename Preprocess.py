import sys
import numpy as np
import scipy.io.wavfile as wavfile
import audioread
import os
import soundfile as sf
from pydub import AudioSegment
from pydub.utils import make_chunks
import librosa
import librosa.display
import matplotlib.pyplot as plt
from numpy import random
#import cv2
from tqdm import tqdm



class RemoveSilence:

    
    # ------------ Beginning of remove_silence() method ---------------------------------------------
    
    def remove_silence(self, fs, signal , frame_duration = 0.022,frame_shift = 0.01, perc = 0.02):

        orig_dtype = type(signal[0])
        typeinfo = np.iinfo(orig_dtype)
        is_unsigned = typeinfo.min >= 0
        signal = signal.astype(np.int64)
        if is_unsigned:
            signal = signal - (typeinfo.max + 1) / 2

        siglen = len(signal)
        retsig = np.zeros(siglen, dtype=np.int64)
        frame_length = int(frame_duration * fs)
        frame_shift_length = int(frame_shift * fs)
        new_siglen = 0
        i = 0
        # NOTE: signal ** 2 where signal is a numpy array
        #       interpret an unsigned integer as signed integer,
        #       e.g, if dtype is uint8, then
        #           [128, 127, 129] ** 2 = [0, 1, 1]
        #       so the energy of the signal is somewhat
        #       right
        average_energy = np.sum(signal ** 2) / float(siglen)
        print
        average_energy
        # print "Avg Energy: ", average_energy
        while i < siglen:
            subsig = signal[i:i + frame_length]
            ave_energy = np.sum(subsig ** 2) / float(len(subsig))
            if ave_energy < average_energy * perc:
                i += frame_length
            else:
                sigaddlen = min(frame_shift_length, len(subsig))
                retsig[new_siglen:new_siglen + sigaddlen] = subsig[:sigaddlen]
                new_siglen += sigaddlen
                i += frame_shift_length
        retsig = retsig[:new_siglen]
        if is_unsigned:
            retsig = retsig + typeinfo.max / 2
        return retsig.astype(orig_dtype)
    
    #-------------- End of remove_silence() method --------------------------------------------------------

    
    
    #--------------- Beginning of __init__() method -------------------------------------------------------
    
    def __init__(self, sample_rate, signal_values, label):

        self. sample_rate = sample_rate
        self.signal_values = signal_values
        self.label = label

        signal_out = self.remove_silence(self.sample_rate, self.signal_values)
        wavfile.write(self.label, sample_rate, signal_out)
    
    #------------  End of __init__() method ----------------------------------------------------------------
  

print("Getting audio path locations ...")

target_loc = "/home/divyas/Workspace/vox2/dev_wav"
audio_path = {}

for dirpath , dir , files in os.walk(target_loc):
    for file in files:
        #speaker = file.split('____')[0]
        #folder = file.split('____')[1]
        #file_name = file.split('____')[2]
        #file_name = file_name.split('.')[0]
        audio_path[file] = dirpath+"/"+file

        
print("Removing silent parts")       
        
# Remove Silent parts
for label in tqdm(audio_path):
    path_to_file = audio_path[label]
    #print(path_to_file)
    sample_rate, signal_values = wavfile.read(path_to_file)
    remove_silence = RemoveSilence(sample_rate, signal_values, '/home/divyas/Workspace/vox2/dev_initial_preprocessed/'+label) 

print("Clip to 3 second chunks")
    
# Clip to 3 second
for label in tqdm(audio_path):    
    path_to_wav = '/home/divyas/Workspace/vox2/dev_initial_preprocessed/'+label
    sample_rate, signal_values = wavfile.read(path_to_wav)
    myaudio = AudioSegment.from_file(path_to_wav , "wav")
    chunk_length_ms = 3000 # Pydub calculates in ms ... change to 3000 to 3 seconds
    chunks = make_chunks(myaudio , chunk_length_ms)
    # Export all the individual chunks as wavfiles
    temp = label.split('.')
    temp = temp[0]
    num = 1
    for i,chunk in enumerate(chunks):
        name = "/home/divyas/Workspace/vox2/ThreeSecond/chunks/dev/"+temp+"____part____"+str(num)+".wav"
        num = num+1
        chunk_name = name.format(i)
        chunk.export(chunk_name, format = "wav")
        f = sf.SoundFile(name)
        time = format(len(f)/f.samplerate)
        if float(time)<3.0: # Change to 3.0 for 3 seconds
            os.unlink(name)
            

