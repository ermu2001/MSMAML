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

ptr = 70
pvl = 15
pts = 15

def load_audio(file_path):
  au, sr = librosa.load(os.path.join(audio_dir, file_path), sr=44100)
  return au.reshape(-1, 1).T

data = []
label = []
file = []

for row in tqdm(meta_df.itertuples(index=True, name='Pandas'), total=len(meta_df)):
    data.append(load_audio(row[1]))
    label.append(row[3])
    file.append(row[1])

dtr = data[:num_sample * ptr]
dvl = data[num_sample * ptr:num_sample * (ptr + pvl)]
dts = data[num_sample * (ptr + pvl):]

ltr = label[:num_sample * ptr]
lvl = label[num_sample * ptr:num_sample * (ptr + pvl)]
lts = label[num_sample * (ptr + pvl):]

ftr = file[:num_sample * ptr]
fvl = file[num_sample * ptr:num_sample * (ptr + pvl)]
fts = file[num_sample * (ptr + pvl):]

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

torch.save({'data': data, 'label': labels, 'file': file}, './data/esc50.pth')

# rs = np.random.RandomState(123)
# rs.shuffle(classes)
# num_train, num_val, num_test = [
#     int(float(ratio)/np.sum(train_val_test_ratio)*len(classes))
#     for ratio in train_val_test_ratio]

# classes = {
#     'train': classes[:num_train],
#     'val': classes[num_train:num_train+num_val],
#     'test': classes[num_train+num_val:]
# }

# if not os.path.exists(target_dir):
#     os.makedirs(target_dir)
# for k in classes.keys():
#     target_dir_k = os.path.join(target_dir, k)
#     if not os.path.exists(target_dir_k):
#         os.makedirs(target_dir_k)
#     cmd = ['mv'] + [os.path.join(source_dir, c) for c in classes[k]] + [target_dir_k]
#     subprocess.call(cmd)

# _ids = []

# for root, dirnames, filenames in os.walk(target_dir):
#     for filename in filenames:
#         if filename.endswith(('.jpg', '.webp', '.JPEG', '.png', 'jpeg')):
#             _ids.append(os.path.join(root, filename))

# for path in _ids:
#     try:
#         img = imread(path)
#     except:
#         print(img)
#     if len(img.shape) < 3:
#         print(path)
#         img = np.tile(np.expand_dims(img, axis=-1), [1, 1, 3])
#         imwrite(path, img)
#     else:
#         if img.shape[-1] == 1:
#             print(path)
#             img = np.tile(img, [1, 1, 3])
#             imwrite(path, img)

# # resize images
# cmd = ['python', 'get_dataset_script/resize_dataset.py', './data/bird']
# subprocess.call(cmd)
