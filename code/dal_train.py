import codecs
import os
import re
import math
import pickle 
import json
import math
import random,random
from functools import reduce 
import numpy as np
import pandas as pd
from sklearn.externals import joblib
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
#
from make_model import *
from read_input_data import *
##################################################################
# divide the labeled corpus and unlabeled corpus 
def divide_LU(data, label_ratio):
    data_len = len(data)
    label_data_len = math.floor(data_len * label_ratio)
    idxs = list(range(data_len))
    np.random.shuffle(idxs)
    data = [data[id_] for id_ in idxs]
    data_train_label = data[:label_data_len]
    data_train_unlabel = data[label_data_len:]
    return data_train_label, data_train_unlabel

def evaluation_al(model, data_test, id2label):
	X_test, y_test= ...xxx(data_test) # divide data_test to x and y
    y_pred = model.predict(X_test)
    y_pred_label =  eva_ylabel(y_pred, id2label)
    y_true_label =  eva_ylabel(y_test, id2label)
    labels = pd.Series(np.concatenate(y_true_label)).unique().tolist()
    if 'O' in labels:
        labels.remove('O')
    if '[PAD]' in labels:
        labels.remove('[PAD]')
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    crf_mtr = metrics.flat_classification_report(y_true_label, y_pred_label, labels=sorted_labels, digits=6)
    crf_acc = metrics.flat_accuracy_score(y_true_label, y_pred_label)
    return getScoreDf(crf_mtr), float(crf_acc)

##################################################################
# HPS+DA
def pool_HPSDA(d_L, d_U, data_dev, data_test, corpus_config, pretrain_config,
		is_AL, u_block, alpha_B, ini_pct, step, mu,
        is_direct_DA=True, aug_method='replace', beta_value=0.9, ent_global_dic={}, param=0, confirm_n=100, 
        is_indirect_DA=True):
	K.clear_session()
    B = alpha_B * len(d_L)
    max_step = int(math.ceil((len(d_U) - len(d_L)) / B))        
	model, callbacks_list = build_NER_network(d_L, data_dev, corpus_config, model_config)
    while (step <= max_step) and (len(d_U) > 0):
    	K.clear_session() 
        history_dic = None
        if step == 1: 
            history_dic = loadData(os.path.join(corpus_config["root_file"], corpus_config["output_filedir"],
            	"{}_{}_history.pkl".format(corpus_config["corpus_name"], str(ini_pct)))
            )
        else:
            history_dic = loadData(os.path.join(corpus_config["root_file"], corpus_config["output_filedir"],
            	"{}_{}_{}_history.pkl".format(corpus_config["corpus_name"], is_AL, str(ini_pct)))
            )
        step = history_dic['step'] + 1
        d_U = history_dic['d_U']
        d_L = history_dic['d_L']
        if step!=1 :
            model, callbacks_list = build_NER_network(d_L, data_dev, corpus_config, model_config)
        old_num_d_L = len(d_L)
        ######################################################################
        # use Direct_DA?
        # Direct DA
        if is_direct_DA:
            if aug_method=='replace':
                d_L = d_L + augment_learn(d_L, ent_global_dic, 'replace', param, confirm_n, beta_value)
            if aug_method=='swap':
                d_L = d_L + augment_learn(d_L, {}, 'swap', param, confirm_n, beta_value)
            elif aug_method=='merge':
                d_L = d_L + augment_learn(d_L, ent_global_dic, 'merge', param, confirm_n, beta_value) # 90 
        ######################################################################
        # use AL ?
        # AL
        if is_AL:
            u_U_iter = math.ceil(len(d_U)/u_block)
            block_end = 0
            d_u_2 = []
            al_sta_list = []
            for u in range(u_U_iter):
                block_end = u_block*u+u_block
                d_u = d_U[u_block*u: u_block*u+u_block]
                X_U_al, y_U_al = get_dev_xy(d_u, pretrain_config["seq_len"], pretrain_config["pad_char"], 
                	model_config["label2id"], True, w2id
                	)
                K_in, K_out, crf_x, T, B_d, O_d = ini_prob_by_M(model, X_U_al)
                crf_input_energy, crf_chain_kernel = ini_prob_by_M2(K_in, K_out, crf_x, X_U_al)
                crf_energy, logZ = ini_prob_by_M3(K_in, K_out, crf_input_energy, crf_chain_kernel, crf_x, X_U_al, y_U_al)
                X_L_al, y_L_al = get_dev_xy(d_L_sample, pretrain_config["seq_len"], pretrain_config["pad_char"], 
                	model_config["label2id"], True, w2id
                )
                _, _, crf_x_L, T, _, _ = ini_prob_by_M(model, X_L_al)

                dpp_score = build_dal(crf_x, crf_x_L, logZ, crf_energy, crf_input_energy, crf_chain_kernel, 
 						B_d, O_d, T, step, ini_pct)
                al_sta_list.extend(our_score)
                d_u_2.extend(d_u)
            d_L, d_u = get_al_data(B, np.array(al_sta_list), d_L, d_u_2)
            d_U = d_u + d_U[block_end:]
        ######################################################################
        if step!=1:
            K.clear_session()
        # use indirect_DA ?
        if is_indirect_DA:
        	model, callbacks_list = build_NER_network(d_L, data_dev, corpus_config, model_config)
            model = PAT(model, 'Embedding-Token', 0.5)
        # train model
        model = trainModel(model, d_L, data_dev, model_config["epoch"] , callbacks_list)
        # evaluation 
        evaluation_metric, crf_accuracy = evaluation_al(model, data_test, id2label) 
        #
        history_dic = {}
        history_dic['step'] = step
        history_dic['d_U'] = d_U
        history_dic['d_L'] = d_L
        # get_evaluation_metric
        print(evaluation_metric, crf_accuracy)
        step = step + 1

##################################################################
# get the prediction of backend NER model
def ini_prob_by_M(model, X_U_al):
    K_in = [model.input, K.learning_phase()]
    K_out = model.get_layer('Output')
    last_2_func = K.function(K_in, [model.layers[len(model.layers)-2].output])
    crf_x = last_2_func([X_U_al])[0]
    T = crf_x.shape[1]
    B_d = crf_x.shape[0] 
    O_d = crf_x.shape[2]
    return K_in, K_out, crf_x, T, B_d, O_d

def ini_prob_by_M2(K_in, K_out, crf_x, X_U_al):
    crf_left_boundary = K.function(K_in, [K_out.left_boundary])([X_U_al])[0]
    crf_right_boundary = K.function(K_in, [K_out.right_boundary])([X_U_al])[0]
    crf_kernel = K.function(K_in, [K_out.kernel])([X_U_al])[0]
    crf_bias = K.function(K_in, [K_out.bias])([X_U_al])[0]
    crf_chain_kernel = K.function(K_in, [K_out.chain_kernel])([X_U_al])[0]
    crf_input_energy = np.dot(crf_x,crf_kernel) + crf_bias
    crf_input_energy = K.function(K_in, [K_out.add_boundary_energy(crf_input_energy, None, crf_left_boundary, crf_right_boundary)])([X_U_al])[0]
    return crf_input_energy, crf_chain_kernel

def ini_prob_by_M3(K_in, K_out, crf_input_energy, crf_chain_kernel, crf_x, X_U_al, y_U_al):
    crf_inner_input_energy = np.sum(crf_input_energy * y_U_al, 2) # B * T
    crf_inner_chain_energy = np.sum(np.dot(y_U_al[:, :-1, :], crf_chain_kernel) * y_U_al[:, 1:, :], 2) 
    crf_energy = np.sum(crf_inner_input_energy, -1) + np.sum(crf_inner_chain_energy, -1) 
    logZ = K.function(K_in, [K_out.get_log_normalization_constant(crf_input_energy, None,)])([X_U_al])[0]
    return crf_energy, logZ

def recusion_N(input_energy, chain_energy_recusion, target_val, k, energy_list, argmin_id, N, B_d, O_d):
    if k >= input_energy.shape[1]:
        return target_val,energy_list
    input_energy_t = input_energy[:,k] 
    energy_ = []
    for target_val_t, argmin_id_t in zip(target_val, argmin_id):
        _ = np.expand_dims(target_val_t,-1) 
        energy = input_energy_t + _ 
        if k!=0:
            energy = chain_energy_recusion[argmin_id_t] + energy 
        energy_.append(energy) 
    energy = np.concatenate(energy_,1) 
    argmin_id2 = []
    for i in range(N):
        x = np.ones((B_d, O_d))
        x = np.cumsum(x,1)-1
        argmin_id2.append(x)
    argmin_id2 = np.concatenate(argmin_id2,1) 
    sort_argmin_id2 = []
    sort_energy = []
    for energy_t, argmin_id2_t in zip(energy, argmin_id2):
        _ = np.sort(energy_t,-1)[:N] 
        sort_energy.append(_) 
        _ = argmin_id2_t[np.argsort(energy_t)][:N]
        _ = _.astype(int)
        sort_argmin_id2.append(_) 
    argmin_id = list(np.array(sort_argmin_id2).T) 
    target_val_2 =  list(np.array(sort_energy).T) 
    energy_list.append(target_val_2)
    return recusion_N(input_energy, chain_energy_recusion, target_val_2, k+1, energy_list, argmin_id, N, B_d, O_d)

def get_al_data(k, d_U_sta, d_L_temp, d_U_temp):
    d_U_temp_idx = np.argsort(-d_U_sta)[:k] 
    d_U_2 = []
    for _i in range(len(d_U_temp)):
        if _i not in d_U_temp_idx: 
            d_U_2.append(d_U_temp[_i])
        else: 
            d_L_temp.append(d_U_temp[_i])
    return d_L_temp, d_U_2

def rank_al_data(k, d_U_sta, d_U_temp):
    d_U_temp_idx = np.argsort(-d_U_sta)[:k] 
    d_U_2 = []
    for _i in d_U_temp_idx:
        d_U_2.append(d_U_temp[_i])
    return d_U_2

def shannon_entropy(arr, epsilon = 1e-20):
    return -arr * np.log(arr+epsilon)

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))

def get_least_confidence(crf_energy, logZ):
    nloglik = crf_energy + logZ
    al_result = 1 - np.exp(-nloglik)
    return al_result

def build_dal(crf_x, crf_x_L, logZ, crf_energy, crf_input_energy, crf_chain_kernel, 
	B_d, O_d, T, step, mu): 
    def GSM_DPP(X_u_cnt, al_selection_size, al_kernel_matrix, epsilon=1e-10):
	    c = np.zeros((al_selection_size, X_u_cnt)) 
	    d = np.copy(np.diag(al_kernel_matrix))
	    j = np.argmax(d)
	    Yg = [j]
	    Yg_score = [np.max(d)]
	    iter = 0
	    Z = list(range(X_u_cnt))
	    while len(Yg) < al_selection_size:
	        Z_Y = set(Z).difference(set(Yg))
	        for i in Z_Y:
	            if iter == 0:
	                ei = al_kernel_matrix[j, i] / np.sqrt(d[j])
	            else:
	                ei = (al_kernel_matrix[j, i] - np.dot(c[:iter, j], c[:iter, i])) / np.sqrt(d[j])
	            c[iter, i] = ei
	            d[i] = d[i] - ei * ei
	        d[j] = 0
	        j = np.argmax(d)
	        j_score = np.max(d)
	        if d[j] < epsilon:
	            break
	        Yg.append(j)
	        Yg_score.append(j_score)
	        iter += 1
	    return Yg,Yg_score
    N = 10
    # PMU
    crf_prev_target_val2= [np.zeros((B_d))] 
    crf_argmin_id = [np.zeros(B_d)] 
    crf_E2, _ = recusion_N(crf_input_energy, crf_chain_kernel, crf_prev_target_val2, 0, [], crf_argmin_id,  N, B_d, O_d) 
    crf_E_list = []
    for E_ in crf_E2:
        neg_log_like_ = -(logZ + E_) 
        p_y_x_ = np.exp(neg_log_like_) 
        crf_E_list.append(p_y_x_)
    p_y_x_2 = np.array(crf_E_list).T 
    SE_ = np.sum(shannon_entropy(p_y_x_2),1) 
    LC = get_least_confidence(crf_energy, logZ)
    pmu_score = np.sqrt(SE/np.log(N)) * np.sqrt(LC)
    #
	x_encode = np.sum(crf_x, 1) 
    x_encode_L = np.sum(crf_x_L, 1) 
    encode_score =  []
    x_encode_mo = np.linalg.norm(x_encode_L, axis=-1)  
    for x_i in x_encode:
        sim_value = np.sum(np.expand_dims(x_i,0) * x_encode_L, -1)
        sim_value = sim_value/(np.linalg.norm(x_i) * x_encode_mo) 
        sim_value = np.max(sim_value) 
        encode_score.append(sim_value)
    encode_score = np.array(encode_score) 
    PLP_sita = math.pow(mu, step)
    pmr_mmr_score = pmu_score * PLP_sita - (1-PLP_sita) * (sigmoid(encode_score))
    #
    x_encode = np.sum(crf_x, 1) 
    x_encode = x_encode / np.linalg.norm(x_encode, axis=1, keepdims=True)
    X_U_al_sim_matrix = np.dot(x_encode, x_encode.T) 
    X_U_al_sim_matrix = (X_U_al_sim_matrix + 1)/2
    x_encode_L = None
    pmr_dpp_score = None
    bkm_kernel_matrix = pmr_mmr_score.reshape((B_d, 1)) * X_U_al_sim_matrix * pmr_mmr_score.reshape((1, B_d))
    dpp, pmr_dpp_score = GSM_DPP(B_d, 1000, bkm_kernel_matrix, epsilon=1e-10)
    pmr_dpp_score = pmr_dpp_score + [0 for i in range(B_d-len(pmr_dpp_score))]
    pmr_dpp_score = np.array(pmr_dpp_score)
    return pmr_dpp_score 
    
##################################################################
# three types of DA 
def augment_learn(data_L, ent_global_dic, method, param, confirm_n, beta_func_alpha):
    if method=='replace':
        return eda(data_L, ent_global_dic, param, 0, 0, beta_func_alpha, confirm_n)
    elif method=='swap':
        return eda(data_L, ent_global_dic, 0, param, 0, beta_func_alpha, confirm_n)
    else:
        return eda(data_L, ent_global_dic, 0, 0, param, beta_func_alpha, confirm_n)

def eda(sentences, ent_global_dic, r_n=100, s_n=100, auto_n=0, beta_func_alpha=0.9, confirm_n=100): 
    augmented_sentences = []
    if auto_n==0:
        if r_n > 0:
            for s in sentences:
                aug_replace_sample_list = replace_n(s, ent_global_dic, r_n)
                augmented_sentences.extend(aug_replace_sample_list) 
        if s_n > 0:
            for s in sentences:
                aug_swap_sample_list = swap_n(s, s_n)
                augmented_sentences.extend(aug_swap_sample_list)
    if auto_n > 0:
        delta = np.random.beta(beta_func_alpha,beta_func_alpha)
        delta = np.max([delta, 1-delta])
        s_n = math.floor(auto_n * delta)
        for s in sentences:
            aug_swap_sample_list = swap_n(s, s_n)
            augmented_sentences.extend(aug_swap_sample_list)
        r_n  = auto_n - s_n
        for s in sentences:
            aug_replace_sample_list = replace_n(s, ent_global_dic, r_n)
            augmented_sentences.extend(aug_swap_sample_list) 
    random.shuffle(augmented_sentences)  
    return augmented_sentences[:confirm_n]

def swap_n(dat_old, n):
	def swap(dat, a_ent_list, b_ent_list):
	    temp = None
	    if a_ent_list[0] > b_ent_list[0]:
	        temp = a_ent_list  
	        a_ent_list = b_ent_list 
	        b_ent_list = temp

	    x_ = dat[0]
	    y_ = dat[1]
	    b_end = 0
	    a_end = 0
	    if len(b_ent_list)==1:
	        b_end = b_ent_list[0]+1
	    else:
	        b_end = b_ent_list[1]+1
	    
	    if len(a_ent_list)==1:
	        a_end = a_ent_list[0]+1
	    else:
	        a_end = a_ent_list[1]+1    
	    
	    x_2 = x_[:a_ent_list[0]] + x_[b_ent_list[0]:b_end] + x_[a_end:b_ent_list[0]] + x_[a_ent_list[0]:a_end] + x_[b_end:]
	    y_2 = y_[:a_ent_list[0]] + y_[b_ent_list[0]:b_end] + y_[a_end:b_ent_list[0]] + y_[a_ent_list[0]:a_end] + y_[b_end:]
	    return [x_2, y_2]
	def sta_ent_idx(sample):
		def filter_ent(ent_name, sta_dic):
		    if len(sta_dic[ent_name])>=2:
		        return True
		    else:
		        return False
	    y_ = sample[1]
	    x_ = sample[0]
	    ent = []
	    ent_name = ''
	    result_dic = {}
	    for idx, y_label in enumerate(y_):
	        _ = id2label[y_label]
	        if _ not in ('O', '[PAD]'):
	            _name = _.split('-')[0]
	            _tag = _.split('-')[1]
	            if _tag == 'B':
	                ent.append(idx)
	                ent_name = _name
	            elif _tag == 'E':
	                ent.append(idx)
	                if _name not in result_dic:
	                    result_dic[_name] = []
	                result_dic[_name].append(ent)
	                ent = []
	                ent_name = ''
	            elif _tag == 'S':  
	                ent.append(idx)
	                if _name not in result_dic:
	                    result_dic[_name] = []
	                result_dic[_name].append(ent)
	        else:
	            ent = []
	            ent_name = '' 
	    select_keys_list = []
	    for k,v in result_dic.items():
	        if filter_ent(k, result_dic):
	            select_keys_list.append(k)
	    return result_dic, select_keys_list

    aug_list = []
    dat_new = dat_old.copy()
    t = 1
    while len(aug_list) <= n:
        if t > 10000:
            return []
        result_dic, select_keys_list =sta_ent_idx(dat_new)
        if len(result_dic.keys())==0:
            return []
        if len(select_keys_list)==0:
            return []
        ent_ = random.sample(select_keys_list,1)[0]
        if len(result_dic[ent_])<2:
            continue
        idx_list = random.sample(result_dic[ent_],2)
        dat_new = swap(dat_new, idx_list[0], idx_list[1])
        if dat_new not in aug_list:
            aug_list.append(dat_new)
        t = t + 1
    return aug_list

def rep(dat, a_ent_list, new_entity, ent_name, label2id):
    new_ent_label = []
    if len(new_entity)==1:
        new_ent_label.append(label2id[ent_name + '-S'])
    elif len(new_entity)==2:
        new_ent_label.append(label2id[ent_name + '-B'])
        new_ent_label.append(label2id[ent_name + '-E'])
    elif len(new_entity)>=3:
        new_ent_label.append(label2id[ent_name + '-B'])
        for i in range(len(new_entity)-2):
            new_ent_label.append(label2id[ent_name + '-I'])
        new_ent_label.append(label2id[ent_name + '-E'])
    x_ = dat[0]
    y_ = dat[1]
    a_end = 0
    if len(a_ent_list)==1:
        a_end = a_ent_list[0]+1
    else:
        a_end = a_ent_list[1]+1    
    x_2 = x_[:a_ent_list[0]] + [_ for _ in new_entity] + x_[a_end:]
    y_2 = y_[:a_ent_list[0]] + new_ent_label + y_[a_end:]
    return [x_2, y_2]


def replace_n(dat_old, ent_global_dic, n):
    aug_list = []
    dat_new = dat_old.copy()
    t = 1
    while len(aug_list) <= n: 
        if t > 10000:
            return []
        result_dic, a =sta_ent_idx(dat_new)
        if len(result_dic.keys())==0:
            return []
        ent_ = random.sample(list(result_dic.keys()),1)[0]
        if len(result_dic[ent_])<1:
            continue
        idx_list = random.sample(result_dic[ent_],1)[0]
        new_entity = random.sample(ent_global_dic[ent_],1)[0]
        ent_name = ent_
        dat_new = rep(dat_new, idx_list, new_entity, ent_, label2id)
        if dat_new not in aug_list:
            aug_list.append(dat_new)
        t = t + 1
    dat_new = dat_old.copy()    
    return aug_list

def get_ent_dic(dat):
    result_dic = {}
    for i in range(len(dat)):
        y_ = dat[i][1]
        x_ = dat[i][0]
        ent = []
        ent_name = ''
        for idx, y_label in enumerate(y_):
            _ = id2label[y_label]
            if _ not in ('O', '[PAD]'):
                _name = _.split('-')[0]
                _tag = _.split('-')[1]
                if _tag == 'B':
                    ent.append(x_[idx])
                    ent_name = _name
                elif _tag == 'E':
                    ent.append(x_[idx])
                    if _name not in result_dic:
                        result_dic[_name] = []
                    result_dic[_name].append(ent)
                    ent = []
                    ent_name = ''
                elif _tag == 'S':  
                    ent.append(x_[idx])
                    if _name not in result_dic:
                        result_dic[_name] = []
                    result_dic[_name].append(ent)
                else:
                    ent.append(x_[idx])
            else:
                ent = []
                ent_name = ''  
    result_dic2 = {}
    for name, v in result_dic.items():
        result_dic2[name]=[''.join(i) for i in v]              
    return result_dic2
##################################################################
# Indirect DA: PAT
def PAT(model, embedding_name, epsilon=1): 
	def search_layer(inputs, name, exclude=None):
	    if exclude is None:
	        exclude = set()
	    if isinstance(inputs, keras.layers.Layer):
	        layer = inputs
	    else:
	        layer = inputs._keras_history[0]

	    if layer.name == name:
	        return layer
	    elif layer in exclude:
	        return None
	    else:
	        exclude.add(layer)
	        inbound_layers = layer._inbound_nodes[0].inbound_layers
	        if not isinstance(inbound_layers, list):
	            inbound_layers = [inbound_layers]
	        if len(inbound_layers) > 0:
	            for layer in inbound_layers:
	                layer = search_layer(layer, name, exclude)
	                if layer is not None:
	                    return layer
    if model.train_function is None:  
        model._make_train_function() 
    old_train_function = model.train_function  

    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')
    embeddings = embedding_layer.embeddings  
    gradients = K.gradients(model.total_loss, [embeddings])  
    gradients = K.zeros_like(embeddings) + gradients[0]  
    inputs = (model._feed_inputs +
              model._feed_targets +
              model._feed_sample_weights)  
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  
    def train_function(inputs):
        grads = embedding_gradients(inputs)[0]  
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  
        K.set_value(embeddings, K.eval(embeddings) + delta) 
        outputs = old_train_function(inputs)  
        K.set_value(embeddings, K.eval(embeddings) - delta)  
        return outputs
    model.train_function = train_function
    return model


##################################################################
pool_HPSDA(d_L, d_U, data_dev, data_test, corpus_config, pretrain_config,
		is_AL=True, 1000, 2, 0.01, 0,
        id2label, eta=1, types="Gaussian", alpha=0.5, train_combine=True, delta=0.95, 
        is_direct_DA=True, aug_method='replace', beta_value=0.9, ent_global_dic=ent_global_dic, param=300, confirm_n=1085, 
        is_indirect_DA=True
)


ini_pct = 0.01
alpha_B = 2
mu = 0.95
ent_global_dic = get_ent_dic(data_train) 
ent_global_dic. update(data_dev)
ent_global_dic.update(data_test)

id2label  = {v:k for k,v in model_config["label2id"].items()}
#
data_L, data_U = divide_LU(data_train, pct)
data_U = data_U[: int(np.floor(len(data_train) * 0.5)) ] 
#
data_dev = get_random_size_of_dataset(data_dev)
data_test =  get_random_size_of_dataset(data_test)
#
pool_HPSDA(data_L, data_U, data_dev, data_test, corpus_config, pretrain_config,
		is_AL=True, u_block=1000, alpha_B=alpha_B, ini_pct=ini_pct, mu=mu, step=0,
        is_direct_DA=True, aug_method='replace', beta_value=0.9, ent_global_dic=ent_global_dic, param=300, confirm_n=1085, 
        is_indirect_DA=True
)
