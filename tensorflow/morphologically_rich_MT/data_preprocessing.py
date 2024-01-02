import keras_nlp
import pathlib
import random

import keras
import keras_nlp

import tensorflow.data as tf_data
from tensorflow_text.tools.wordpiece_vocab import (
    bert_vocab_from_dataset as bert_vocab,
)

import os 
import glob
import re

import nltk

import numpy as np

from gensim.models import Word2Vec

np.random.seed(10)

BATCH_SIZE = 64
EPOCHS = 200  # This should be at least 10 for convergence
MAX_SEQUENCE_LENGTH = 40
ENG_VOCAB_SIZE = 75000
TAM_VOCAB_SIZE = 350000

EMBED_DIM = 256
INTERMEDIATE_DIM = 2048
NUM_HEADS = 8

DATASET_DIR = "dataset/en-ta-parallel-v2"
file_names = lambda split, lang: "corpus.bcn.%s.%s" % (split, lang)

def get_sentence_pairs(split):

    text_pairs = []
    with open(os.path.join(DATASET_DIR, file_names(split, "en")), 'r') as l1:
        with open(os.path.join(DATASET_DIR, file_names(split, "ta")), 'r') as l2:
            
            eng_sentences = [re.sub(
                                '\d+', ' [NUM] ', re.sub( # replace all numbers with [NUM] token
                                    r'([^\w\s]|_)','', x.lower()) # remove all symbols
                                ).strip().replace("  ", " ")  
                                        for x in l1.readlines()]
            
            tam_sentences = [re.sub(
                                '\d+', ' [NUM] ', re.sub( # replace all numbers with [NUM] token
                                    r'([^\w\s]|_)','', x.lower()) # remove all symbols
                                ).strip().replace("  ", " ")  
                                        for x in l2.readlines()] # some english words show up in tamil dataset (lower case)

    for eng, tam in zip(eng_sentences, tam_sentences):
        text_pairs.append((eng, tam))
    
    random.shuffle(text_pairs)
    return text_pairs

train_pairs = get_sentence_pairs("train")
val_pairs = get_sentence_pairs("dev")
test_pairs = get_sentence_pairs("test")
text_pairs = train_pairs + val_pairs + test_pairs

num_train_samples = len(train_pairs)
num_val_samples = len(val_pairs)
num_test_samples = len(test_pairs)

print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")

def return_english_word2vec(tokens, sentences, word_vector_size=100):

    if not os.path.isdir('word2vec'):
        os.mkdir('word2vec')

    if not os.path.exists("word2vec/vocab%d_%d.EN" % (len(tokens), word_vector_size)):
        print("     Creating and Storing Word2Vec vectors for English")
        
        ewv=[]
        for sentence in sentences:
            sent=[]
            for p in sentence.split(" "):
                sent.append(p)
            if not sent == []:
                ewv.append(sent)
        
        modeleng = Word2Vec(ewv, vector_size=word_vector_size, window=5, workers=4, batch_words=50, min_count=1)
        modeleng.save("word2vec/vocab%d_%d.EN" % (len(tokens), word_vector_size))
        
        print("     English Word2Vec model created and saved successfully!")
        
    modeleng = Word2Vec.load("word2vec/vocab%d_%d.EN" % (len(tokens), word_vector_size))
    vec = np.array([modeleng.wv[x] for x in tokens])

    return vec

def return_tamil_word2vec(tokens, sentences, word_vector_size=100):
    
    if not os.path.isdir('word2vec'):
        os.mkdir('word2vec')
    
    if not os.path.exists("word2vec/vocab%d_%d.TA" % (len(tokens), word_vector_size)):
        
        print("     Creating Word2Vec vectors for Tamil")
        
        twv=[]
        for sentence in sentences:
            sent=[]
            for p in sentence.split(" "):
                sent.append(p)
            if not sent == []:
                twv.append(sent)
        
        modeltam = wv.Word2Vec(twv, size=word_vector_size, window=5, workers=4, batch_words=50, min_count=1)
        modeltam.save("word2vec/vocab%d_%d.TA" % (len(vocab), word_vector_size))
        
        print("     Word2Vec model created and saved successfully!")
    
    modeltam = Word2Vec.load("word2vec/vocab%d_%d.TA" % (len(tokens), word_vector_size))
    vec = np.array([modeltam.wv[x] for x in sentences])
    print (modeltam.most_similar(""))
    
    return vec

reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]", "[NUM]"]
for _ in range(500):
    string="%s " % reserved_tokens[2]
    string += ("%s " % reserved_tokens[1]) * np.random.randint(0,3)
    string += ("%s " % reserved_tokens[4]) * np.random.randint(0,3)
    string += ("%s " % reserved_tokens[4]) * np.random.randint(0,3)
    string += ("%s " % reserved_tokens[1]) * np.random.randint(0,3)
    string += ("%s " % reserved_tokens[1]) * np.random.randint(0,3)
    string += ("%s " % reserved_tokens[4]) * np.random.randint(0,3)
    string += "%s" % reserved_tokens[3]
    string += ("%s " % reserved_tokens[0]) * np.random.randint(0,3)
    string += ("%s " % reserved_tokens[0]) * np.random.randint(0,3)
    string += ("%s " % reserved_tokens[0]) * np.random.randint(0,3)

    train_pairs.append((string, string))

eng_samples = [text_pair[0] for text_pair in train_pairs]
 
# 149309 before tokenization ; 75210 after
# 70765 tokens without symbols
# 67016 tokens without numbers
eng_vocab = set()

for idx, sentence in enumerate(eng_samples):
    if idx == len(eng_samples) - 500:
        eng_vocab.update(reserved_tokens)
        break
    else:
        tokens = nltk.word_tokenize(sentence) 
        eng_vocab.update(tokens)
        eng_samples[idx] = " ".join(tokens)

tam_samples = [text_pair[1] for text_pair in train_pairs]

p = re.compile("([.,!?\"':;)(])")
q = re.compile("\s(\s)*")

for idx, sentence in enumerate(tam_samples[:len(sentence) - 500]): # for reserved tokens
    sentence = re.sub(p, "", sentence)
    sentence = re.sub(q, " ", sentence)

    tam_samples[idx] = sentence

#for idx, sentence in enumerate(tam_samples):
#    if idx == len(tam_samples) - 500:
#        tam_vocab.update(reserved_tokens)
#        break
#    else:
#        tokens = nltk.word_tokenize(sentence) 
#        eng_vocab.update(tokens)
#        eng_samples[idx] = " ".join(tokens)

#tam_vocab = train_word_piece(tam_samples, TAM_VOCAB_SIZE, reserved_tokens)

print("English Tokens: ", len(eng_vocab))
#print("Tamil Tokens: ", len(tam_vocab))

english_word_vectors = return_english_word2vec(eng_vocab, eng_samples)
#tamil_word_vectors = return_tamil_word2vec(tam_vocab, tam_samples)
