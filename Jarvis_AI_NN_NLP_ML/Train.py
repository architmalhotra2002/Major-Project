import numpy as np 
import json
import torch 
import torch.nn as nn 
from  torch.utils.data import dataset,dataloader
from  NeuralNetwork import bag_of_words , tokenize , stem
from Brain import NeuralNet

with open('intents.json','r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        print(pattern)
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))


