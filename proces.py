from os import listdir
from numpy import array
from numpy import argmax
from pandas import DataFrame
from nltk.translate.bleu_score import corpus_bleu
from pickle import load
import numpy
from numpy import *
from shutil import copy

def load_set(filename):
        doc = load_doc(filename)
        dataset = list()
        # process line by line
        for line in doc.split('\n'):
                # skip empty lines
                if len(line) < 1:
                        continue
                # get the image identifier
                identifier =line
                dataset.append(identifier)
        return set(dataset)
def load_doc(filename):
        # open the file as read only
        file = open(filename, 'r')
        # read all text
        text = file.read()
        # close the file
        file.close()
        return text

filename = 'Flickr_8k.trainImages.txt'
train= load_set(filename)
for name in train:
	copy('/home/lakshminarasimhan/Projectimages/Flicker8k_Dataset/'+name,'/home/lakshminarasimhan/final/train100/'+name)	
