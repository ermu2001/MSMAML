import numpy as np
import os
import pandas as pd
import librosa
from tqdm import tqdm
import torch
from sklearn.utils import shuffle


audio_dir = './data/ESC-50/audio'
metadata = './data/ESC-50/meta/esc50.csv'
meta_df = pd.read_csv(metadata)
meta_df = shuffle(meta_df)
num_sample = len(meta_df)

ptr = 0.7
pvl = 0.15
pts = 0.15

def load_audio(file_path):
  au, sr = librosa.load(os.path.join(audio_dir, file_path), sr=44100)
  return au.reshape(-1, 1).T

data = []
label = []
file = []
classmap = {}

for row in tqdm(meta_df.itertuples(index=True, name='Pandas'), total=len(meta_df)):
    data.append(load_audio(row[1]))
    label.append(row[3])
    file.append(row[1])
    classmap[int(row[3])] = row[4]
    
tr_split = int(num_sample * ptr)
vl_split = int(num_sample * (ptr + pvl))

dtr = data[:tr_split]
dvl = data[tr_split:vl_split]
dts = data[vl_split:]

ltr = label[:tr_split]
lvl = label[tr_split:vl_split]
lts = label[vl_split:]

ftr = file[:tr_split]
fvl = file[tr_split:vl_split]
fts = file[vl_split:]

labels = {
  "train": ltr, 
  "val": lvl, 
  "test": lts}

data = {  
  "train": dtr, 
  "val": dvl, 
  "test": dts}

file = {
  "train": ftr, 
  "val": fvl, 
  "test": fts}

torch.save({'data': data, 'label': labels, 'file': file, 'class': classmap}, './data/esc50.pth')