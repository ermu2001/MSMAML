# import nltk
# from nltk.corpus import brown
# nltk.download('brown')
# categories = brown.categories()

# texts = []
# categories = []
# vectors = []
# categories_map = {}
# ptr = 0.7
# pvl = 0.15
# pts = 0.15

# for idx, category in enumerate(brown.categories()):
#     categories_map[idx] = category
#     for fileid in brown.fileids(categories=category):
#         text = ' '.join(brown.words(fileids=fileid))
#         texts.append(text)
#         categories.append(idx)

import nltk
from nltk.corpus import brown
import gensim
from gensim.models import Word2Vec
import random
import numpy as np
import torch

nltk.download('brown')
nltk.download('punkt')

texts = []
tok_texts = []
categories = []
vectors = []
categories_map = {}

ptr = 0.7
pvl = 0.15
pts = 0.15

for idx, category in enumerate(brown.categories()):
    categories_map[idx] = category
    for fileid in brown.fileids(categories=category):
        words = brown.words(fileids=fileid)
        text = ' '.join(words)
        tokenized_text = nltk.word_tokenize(text)
        texts.append(text)
        tok_texts.append(tokenized_text)
        categories.append(idx)

seqlen = 3000
vector_size = 128
model = Word2Vec(sentences=tok_texts, vector_size=128, window=5, min_count=5, workers=4)

for text in tok_texts:
    # vector = sum(model.wv[word] for word in text if word in model.wv) / len(text)
    # print(len(text))
    vector = [model.wv[word] for word in text if word in model.wv]
    vector = vector + [np.zeros_like(vector[0])] * (seqlen - len(vector))
    vector = np.stack(vector)
    vectors.append(vector)

vectors = np.stack(vectors).transpose(0, 2, 1)
indices = list(range(len(texts)))
random.shuffle(indices)

texts = [texts[i] for i in indices]
categories = [categories[i] for i in indices]
vectors = vectors[indices]
num_sample = len(texts)
tr_split = int(num_sample * ptr)
vl_split = int(num_sample * (ptr + pvl))

dtr = texts[:tr_split]
dvl = texts[tr_split:vl_split]
dts = texts[vl_split:]

ltr = categories[:tr_split]
lvl = categories[tr_split:vl_split]
lts = categories[vl_split:]

vtr = vectors[:tr_split]
vvl = vectors[tr_split:vl_split]
vts = vectors[vl_split:]

# import pdb; pdb.set_trace()

labels = {
    "train": ltr, 
    "val": lvl, 
    "test": lts}

text = {  
    "train": dtr, 
    "val": dvl, 
    "test": dts}

vector = {
    "train": vtr, 
    "val": vvl, 
    "test": vts}

torch.save({'data': vector, 'label': labels, 'file': text, 'class': categories_map}, './data/brown_sequence_3000.pth')