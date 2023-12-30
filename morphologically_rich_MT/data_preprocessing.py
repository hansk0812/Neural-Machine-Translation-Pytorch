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
            eng_sentences = [x.lower() for x in l1.readlines()]
            tam_sentences = [x.lower() for x in l2.readlines()] # some english words show up in tamil dataset

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

def train_word_piece(text_samples, vocab_size, reserved_tokens):
    word_piece_ds = tf_data.Dataset.from_tensor_slices(text_samples)
    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
        word_piece_ds.batch(1000).prefetch(2),
        vocabulary_size=vocab_size,
        reserved_tokens=reserved_tokens,
    )
    return vocab

reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

eng_samples = [text_pair[0] for text_pair in train_pairs]
eng_vocab = train_word_piece(eng_samples, ENG_VOCAB_SIZE, reserved_tokens)

tam_samples = [text_pair[1] for text_pair in train_pairs]
tam_vocab = train_word_piece(tam_samples, TAM_VOCAB_SIZE, reserved_tokens)

print("English Tokens: ", len(eng_vocab))
print("Spanish Tokens: ", len(spa_vocab))

exit()

eng_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=eng_vocab, lowercase=False
)
spa_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=spa_vocab, lowercase=False
)

eng_input_ex = text_pairs[0][0]
eng_tokens_ex = eng_tokenizer.tokenize(eng_input_ex)
print("English sentence: ", eng_input_ex)
print("Tokens: ", eng_tokens_ex)
print(
    "Recovered text after detokenizing: ",
    eng_tokenizer.detokenize(eng_tokens_ex),
)

print()

spa_input_ex = text_pairs[0][1]
spa_tokens_ex = spa_tokenizer.tokenize(spa_input_ex)
print("Spanish sentence: ", spa_input_ex)
print("Tokens: ", spa_tokens_ex)
print(
    "Recovered text after detokenizing: ",
    spa_tokenizer.detokenize(spa_tokens_ex),
)

