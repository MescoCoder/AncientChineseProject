import os
from keras_bert import Tokenizer

# the path of bert
config_path = os.path.join(root_file, "bert", "chinese_L-12_H-768_A-12/bert_config.json")
checkpoint_path = os.path.join(root_file, "bert", "chinese_L-12_H-768_A-12/bert_model.ckpt")
dict_path = os.path.join(root_file, "bert", "chinese_L-12_H-768_A-12/vocab.txt")
token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
class MyTokenizer(Tokenizer):
    def _tokenize(self, text):
        td = []
        for c in text:
            if c in self._token_dict:
                td.append(c)
            else:
                td.append('[UNK]') 
        return td