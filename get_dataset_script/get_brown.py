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

nltk.download('brown')
nltk.download('punkt')

texts = []
tok_texts = []
categories = []
vectors = []
categories_map = {}

for idx, category in enumerate(brown.categories()):
    categories_map[idx] = category
    for fileid in brown.fileids(categories=category):
        words = brown.words(fileids=fileid)
        text = ' '.join(words)
        tokenized_text = nltk.word_tokenize(text)
        texts.append(text)
        tok_texts.append(tokenized_text)
        categories.append(idx)

model = Word2Vec(sentences=tok_texts, vector_size=2000, window=5, min_count=5, workers=4)

for text in texts:
    vector = sum(model.wv[word] for word in text if word in model.wv) / len(text)
    vectors.append(vector)
