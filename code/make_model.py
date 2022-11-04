import tensorflow as tf
import keras
import keras.backend as K
from gensim.corpora.dictionary import Dictionary
from gensim.models import Word2Vec
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.engine.topology import Layer
from keras.layers import *
from keras.layers import Multiply, Add, Reshape, Lambda, Subtract
from keras.layers import Conv1D, ZeroPadding1D, MaxPool1D, Dense, Flatten, concatenate, Embedding, GlobalMaxPooling1D, Activation
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras import initializers
from keras_contrib.layers import CRF 
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateScheduler,ReduceLROnPlateau
from keras_contrib.losses import crf_loss,crf_nll
from keras_contrib.metrics import crf_accuracy,crf_viterbi_accuracy
from keras.utils import plot_model

# loss function
class LossHistory(keras.callbacks.Callback):
    def __init__(self, file_path):
        self.fw = open(file_path, 'a+', encoding='utf-8', buffering=1)
    def on_epoch_end(self, epoch, logs={}):
        self.fw.write(json.dumps(
            {'epoch': int(epoch), 
            'loss': float(logs.get('loss')), 
            'crf_accuracy':float(logs.get('crf_accuracy')), 
            'val_loss':float(logs.get('val_loss')), 
            'val_crf_accuracy':float(logs.get('val_crf_accuracy'))}) + '\n')       
    def on_train_end(self, logs={}):
        self.fw.close()

# compile the model
def compileModel(model, optimizer, filepath, verborse=1):
    if os.path.exists(filepath):
        model.load_weights(filepath)
        print("Loading the file of {} successfully.".format(filepath))
    else:
        print("This is the initial training without any check point.")
    model.compile(
        loss= crf_loss,
        optimizer= optimizer, 
        metrics=[crf_accuracy]
    )
    if verborse:
        model.summary()
    return model


# train the model
def trainModel(model, train_dg, dev_dg, epoch, callbacks_list):
    model.fit_generator(
        train_dg.__iter__(),
        steps_per_epoch=len(train_dg),
        epochs=epoch,
        validation_data=dev_dg.__iter__(),
        validation_steps=len(dev_dg),
        callbacks=callbacks_list,
        verbose=1
    )
    return model

# contruct the network of NER model
def build_model(pretrain_config, labels_num, dropout_date, bilstm_units, optimizer):
    bert_model = load_trained_model_from_checkpoint(pretrain_config["config_path"], pretrain_config["checkpoint_path"], pretrain_config["seq_len"])
    for l in bert_model.layers:
        l.trainable = True    
    x1_input = Input(shape=(pretrain_config["seq_len"],))
    x2_input = Input(shape=(pretrain_config["seq_len"],))
    x_input = bert_model([x1_input, x2_input])
    hidden_layer = bert_model([x1_in, x2_in])
    rnn_layer=Bidirectional(LSTM(bilstm_units, return_sequences=True))(hidden_layer)
    hidden_layer = Dropout(dropout_date)(rnn_layer, training=True)
    TimeDistributed_layer = TimeDistributed(Dense(labels_num))(hidden_layer)
    output = CRF(labels_num, sparse_target=False,learn_mode='join', test_mode='viterbi', name="Output")(TimeDistributed_layer)
    model = Model([x_input], outputs=[output])
    return model


#
def build_NER_network(data_train, data_test, corpus_config, model_config):
    model_net = build_model(model_config["pretrain_config"] , model_config["labels_num"], 
                model_config["dropout_date"], model_config["labels_num"], 
                model_config["optimizer"]
    )
    weight_best_filepath= os.path.join(corpus_config["root_file"],corpus_config["output_filedir"],
        "{}_{}_{}_weights_best.hdf5".format(corpus_config["corpus_name"], corpus_config["method"],
            corpus_config["al_pct_name"])
    )
    checkpoint = ModelCheckpoint(weight_best_filepath, monitor='val_loss', 
        save_weights_only=True, verbose=model_config["verbose"], 
        save_best_only=True, mode='auto', period=model_config["verbose"]
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, 
        mode='auto', verbose=model_config["verbose"]
    )
    history_assessment = LossHistory(os.path.join(root_file, corpus_config["output_filedir"], "{}_{}_{}_loss_log.txt").format(corpus_config["corpus_name"], corpus_config["method"],
            corpus_config["al_pct_name"]))
    learning_rate = ReduceLROnPlateau(monitor='val_loss', patience=3, mode='auto')
    callbacks_list = [checkpoint, early_stopping, history_assessment, learning_rate]
    
    model_net = compileModel(model_net, model_config["optimizer"], weight_best_filepath, verborse=model_config["verbose"])
    return model_net, callbacks_list

from init_bert import *
from read_input_data import *

#
corpus_config["root_file"] = root_file 
corpus_config["output_filedir"] = "active_learning_output" 
corpus_config["corpus_name"] = corpus_name 
corpus_config["method"] = method
corpus_config["al_pct_name"] = al_pct_name


tokenizer = MyTokenizer(token_dict)   
pretrain_config = {}
pretrain_config["config_path"] = config_path
pretrain_config["checkpoint_path"] = checkpoint_path
pretrain_config["seq_len"] = maxlen
pretrain_config["pad_char"] = "pad_char"
#
epoch = 50
model_config = {}
model_config["pretrain_config"] = pretrain_config
model_config["labels_num"] = len(label2id)
model_config["dropout_date"] = 0.3
model_config["bilstm_units"] = 100
optimizer_value = 1e-5
model_config["optimizer"] = Adam(optimizer_value)
model_config["batch_size"] = 64
model_config["verbose"] = 1
#


model, callbacks_list = build_NER_network(data_train, data_dev, corpus_config, model_config)
trainModel(model, train_dg, valid_dg, epoch, callbacks_list)
