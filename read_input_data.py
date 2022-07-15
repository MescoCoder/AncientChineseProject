import os
import math
import pickle
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

root_file = "......"
cmag_config = pd.read_pickle(open(os.path.join(root_file, "CMAG", "data_cmag_config.pkl", "rb"))
maxlen = cmag_config['maxlen'] # the maximum length of all the sentences
label2id = cmag_config['label2id'] # a dictionary for the entity tag schema

data_train = joblib.load(open(os.path.join(root_file,"CMAG", "data_cmag_train.pkl", "rb" ))
data_dev = joblib.load(open(os.path.join(root_file,"CMAG", "data_cmag_dev.pkl", "rb" ))
# for training and parameter adjustment
data_train = data_generator(data_train, .....)
data_dev = data_generator(data_dev, .....)

# for evaluation
data_test = joblib.load(open(os.path.join(root_file,"dataset", "data_cmag_test.pkl", "rb" ))

'''
    Function: a data generator for the training of large batches of data streams
    Input: data, the size of batch, the maximum length for sentences, a string-type sign for padding (defaut: "[PAD]"), a tag schema dictionary
    Output: a data generator object
'''
class data_generator:
    def __init__(self, data, batch_size, maxlen, pad_char, label2id):
        self.maxlen = maxlen
        self.data = data
        self.batch_size = batch_size
        self.steps = math.ceil(len(self.data) / self.batch_size)
        self.pad_char = pad_char
        self.label2id = label2id

    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, Y = [], []
            for idx_ in idxs:
                d = self.data[idx_]  
                text = d[0][:self.maxlen] 
                x1 = ...xx(text) # Transform the words to ids for x 
                y = d[1]
                X1.append(x1)
                y = ...xx(y) # Transform y to the onehot categorical forms
                Y.append()

                if len(X1) == self.batch_size or idx_ == idxs[-1]:
                    X1 = pad_sequences(X1, self.maxlen, padding='post').tolist()
                    yield [np.array(X1)], np.array(Y)
                    [X1, Y] = [], []




