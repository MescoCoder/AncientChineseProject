# AncientChineseProject
A NER dataset for ancient Chinese

# Instructions
Anyone can use the Python script "read_input_data.py" to obtain data from the dataset. But some codes are omitted and need to be modified according to users' own model configuration and enviroment. 

# Code
The code is for Deep active learning (Hybrid Pooled Strategy, HPS) and data augmentation (DA) in NER taks, which include 4 parts:
(1) read_input_data.py: obtain the data
(2) init_bert.py: get the pretrained model "BERT"
(3) make_model.py: construct BERT-BLSTM-CRF
(4) dal_train.py: use the HPS+DA approach to train model

Some extra exsting data like BERT are not provided, one can download it from google-BERT (https://github.com/google-research/bert) or Siku-BERT (https://github.com/hsc748NLP/SikuBERT-for-digital-humanities-and-classical-Chinese-information-processing)

