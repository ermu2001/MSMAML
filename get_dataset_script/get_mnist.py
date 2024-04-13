import numpy as np
import gzip
import os
import requests
import torch

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 28, 28)
    return data

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels

def download_file(url, local_filename):
    """ Helper function to download a file from a given URL """
    response = requests.get(url)
    with open(local_filename, 'wb') as f:
        f.write(response.content)
    return local_filename

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 28, 28)
    return data

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

directory_path = "./data/mnist"
ensure_directory(directory_path)

base_url = "http://yann.lecun.com/exdb/mnist/"
files = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]

for file in files:
    local_filename = os.path.join("./data/mnist", file)
    if not os.path.exists(local_filename):
        print(f"Downloading {file}...")
        download_file(base_url + file, local_filename)

dtr = load_mnist_images(os.path.join(directory_path, "train-images-idx3-ubyte.gz"))
ltr = load_mnist_labels(os.path.join(directory_path, "train-labels-idx1-ubyte.gz"))
dts = load_mnist_images(os.path.join(directory_path, "t10k-images-idx3-ubyte.gz"))
lts = load_mnist_labels(os.path.join(directory_path, "t10k-labels-idx1-ubyte.gz"))

print("Data loaded successfully.")

labels = {
    "train": ltr, 
    "test": lts}

data = {  
    "train": dtr, 
    "test": dts}

torch.save({'data': data, 'label': labels}, './data/mnist.pth')

