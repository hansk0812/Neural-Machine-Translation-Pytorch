import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
from typing import Iterable, List

import random
import re
import numpy as np
import nltk
#nltk.download('punkt')

class EnTamV2Dataset(Dataset):

    SRC_LANGUAGE = 'en'
    TGT_LANGUAGE = 'ta'
    reserved_tokens = ["<unk>", "<pad>", "<start>", "<end>", "<num>"]
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, NUM_IDX = 0, 1, 2, 3, 4
  
    def __init__(self, split):
        
        self.bilingual_pairs = self.get_sentence_pairs(split)
        print ("Using %s set with %d sentence pairs" % (split, len(self.bilingual_pairs)))

        self.eng_vocabulary = self.create_vocabulary([x[0] for x in self.bilingual_pairs], language="en")
        #self.tam_vocabulary = self.create_vocabulary([x[1] for x in self.bilingual_pairs], language="ta")
        
        print ("English vocabulary size for %s set: %d" % (split, len(self.eng_vocabulary)))
        #print ("Tamil vocabulary size for %s set: %d" % (split, len(self.tam_vocabulary)))

    def get_dataset_filename(self, split, lang): 
        assert split in ['train', 'dev', 'test'] and lang in ['en', 'ta']
        return "dataset/corpus.bcn.%s.%s" % (split, lang) 

    def get_sentence_pairs(self, split):

        text_pairs = []
        with open(self.get_dataset_filename(split, self.SRC_LANGUAGE), 'r') as l1:
            with open(self.get_dataset_filename(split, self.TGT_LANGUAGE), 'r') as l2:
                
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
    
    def return_english_word2vec(self, tokens, sentences, word_vector_size=100):

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
    
    def create_token_sentences_for_word2vec(self):

        token_sentences = []
        for _ in range(500):
            string="%s " % self.reserved_tokens[self.BOS_IDX]
            string += ("%s " % self.reserved_tokens[self.PAD_IDX]) * np.random.randint(0,3)
            string += ("%s " % self.reserved_tokens[self.NUM_IDX]) * np.random.randint(0,3)
            string += ("%s " % self.reserved_tokens[self.UNK_IDX]) * np.random.randint(0,3)
            string += ("%s " % self.reserved_tokens[self.NUM_IDX]) * np.random.randint(0,3)
            string += ("%s " % self.reserved_tokens[self.UNK_IDX]) * np.random.randint(0,3)
            string += ("%s " % self.reserved_tokens[self.UNK_IDX]) * np.random.randint(0,3)
            string += "%s" % self.reserved_tokens[self.EOS_IDX]
            string += ("%s " % self.reserved_tokens[self.PAD_IDX]) * np.random.randint(0,3)
            string += ("%s " % self.reserved_tokens[self.PAD_IDX]) * np.random.randint(0,3)
            string += ("%s " % self.reserved_tokens[self.PAD_IDX]) * np.random.randint(0,3)

            token_sentences.append((string, string))

        return token_sentences

    def create_vocabulary(self, sentences, language):
        
        assert language in ['en', 'ta']

        for idx, sentence in enumerate(sentences):
            sentences[idx] = self.reserved_tokens[self.BOS_IDX] + ' ' + sentence + ' ' + self.reserved_tokens[self.EOS_IDX]
        sentences.append(self.create_token_sentences_for_word2vec())
        
        if language == "en":
            # English
            # 149309 before tokenization ; 75210 after
            # 70765 tokens without symbols
            # 67016 tokens without numbers
            vocab = set()
            
            for idx, sentence in enumerate(sentences):
                if idx == len(sentences) - 500:
                    vocab.update(self.reserved_tokens)
                    break
                else:
                    tokens = nltk.word_tokenize(sentence)
                    vocab.update(tokens)
                    sentences[idx] = " ".join(tokens)
        else:
            pass

        return vocab

train_dataset = EnTamV2Dataset("train")
val_dataset = EnTamV2Dataset("dev")
test_dataset = EnTamV2Dataset("test")

exit()

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


# Place-holders
token_transform = {}
vocab_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
# If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)

from torch.nn.utils.rnn import pad_sequence

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

