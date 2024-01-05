import string
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
from typing import Iterable, List

import os
import random
import re
import numpy as np

from requests.exceptions import ConnectionError
import nltk
import stanza
try:
    nltk.download('punkt')
    stanza.download('en')
    #stanza.download('ta')
    en_nlp = stanza.Pipeline('en', processors='tokenize')
    #ta_nlp = stanza.Pipeline('ta', processors='tokenize')
except ConnectionError:
    en_nlp = stanza.Pipeline('en', processors='tokenize', download_method=None)
    #ta_nlp = stanza.Pipeline('ta', processors='tokenize', download_method=None)

from utils.dataset_visualization import visualize_dataset_for_bucketing_stats

class EnTamV2Dataset(Dataset):

    SRC_LANGUAGE = 'en'
    TGT_LANGUAGE = 'ta'
    reserved_tokens = ["UNK", "PAD", "START", "END", "NUM", "ENG"]
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, NUM_IDX, ENG_IDX = 0, 1, 2, 3, 4, 5
  
    def __init__(self, split):
        
        tokenized_dirname = "tokenized"
        if not os.path.exists(self.get_dataset_filename(split, "en", tokenized_dirname)) \
                or not os.path.exists(self.get_dataset_filename(split, "ta", tokenized_dirname)):
            
            self.bilingual_pairs, eng_words = self.get_sentence_pairs(split)
            
            if split == "train":
                eng_words = list(eng_words)
                self.create_token_sentences_for_word2vec(eng_words)

            self.eng_vocabulary, self.eng_word_counts, tokenized_eng_sentences = self.create_vocabulary([
                                                                                    x[0] for x in self.bilingual_pairs], language="en")
            self.tam_vocabulary, self.tam_word_counts, tokenized_tam_sentences = self.create_vocabulary([
                                                                                    x[1] for x in self.bilingual_pairs], language="ta")
            
            print ("English vocabulary size for %s set: %d" % (split, len(self.eng_vocabulary)))
            print ("Tamil vocabulary size for %s set: %d" % (split, len(self.tam_vocabulary)))

            if split == 'train':
                print ("Most Frequent 1000 English tokens:", sorted(self.eng_word_counts, key=lambda y: self.eng_word_counts[y], reverse=True)[:1000])
                print ("Most Frequent 1000 Tamil tokens:", sorted(self.tam_word_counts, key=lambda y: self.tam_word_counts[y], reverse=True)[:1000])

            # save tokenized sentences for faster loading
            save_dir = os.path.dirname(self.get_dataset_filename("", "", tokenized_dirname))
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            with open(self.get_dataset_filename(split, "en", tokenized_dirname), 'w') as f:
                for line in tokenized_eng_sentences:
                    f.write("%s\n" % line)
            with open(self.get_dataset_filename(split, "ta", tokenized_dirname), 'w') as f:
                for line in tokenized_tam_sentences:
                    f.write("%s\n" % line)
            with open(self.get_dataset_filename(split, "en", tokenized_dirname, vocab=True), 'w') as f:
                for word in self.eng_vocabulary:
                    f.write("%s\n" % word)
            with open(self.get_dataset_filename(split, "ta", tokenized_dirname, vocab=True), 'w') as f:
                for word in self.tam_vocabulary:
                    f.write("%s\n" % word)
            self.bilingual_pairs = list(zip(tokenized_eng_sentences, tokenized_tam_sentences))
        
        else:
            
            with open(self.get_dataset_filename(split, "en", tokenized_dirname), 'r') as f:
                tokenized_eng_sentences = [x.strip() for x in f.readlines()]
            with open(self.get_dataset_filename(split, "ta", tokenized_dirname), 'r') as f:
                tokenized_tam_sentences = [x.strip() for x in f.readlines()]
            with open(self.get_dataset_filename(split, "en", tokenized_dirname, vocab=True), 'r') as f:
                self.eng_vocabulary = [x.strip() for x in f.readlines()]
            with open(self.get_dataset_filename(split, "ta", tokenized_dirname, vocab=True), 'r') as f:
                self.tam_vocabulary = [x.strip() for x in f.readlines()]
            self.bilingual_pairs = list(zip(tokenized_eng_sentences, tokenized_tam_sentences))
        
        print ("Using %s set with %d sentence pairs" % (split, len(self.bilingual_pairs)))

        if not os.path.exists('utils/Correlation.png') and split == "train":
            visualize_dataset_for_bucketing_stats(self.bilingual_pairs)
    
    def get_dataset_filename(self, split, lang, subdir=None, vocab=False): 
        assert split in ['train', 'dev', 'test', ''] and lang in ['en', 'ta', ''] # Using '' to get dirname because dataset was defined first here!
        if not subdir is None:
            directory = os.path.join("dataset", subdir, "%s.bcn" % ("corpus" if not vocab else "vocab"))
        else:
            directory = os.path.join("dataset", "%s.bcn" % ("corpus" if not vocab else "vocab"))
        return "%s.%s.%s" % (directory, split, lang) 

    def get_sentence_pairs(self, split):

        text_pairs = []
        translator = str.maketrans('', '', string.punctuation)
        with open(self.get_dataset_filename(split, self.SRC_LANGUAGE), 'r') as l1:
            with open(self.get_dataset_filename(split, self.TGT_LANGUAGE), 'r') as l2:
                
                eng_sentences = [re.sub(
                                    '\d+', ' %s ' % self.reserved_tokens[self.NUM_IDX], re.sub( # replace all numbers with [NUM] token
                                        r'([^\w\s]|_)','', x.lower()) # remove all symbols
                                    ).strip().replace("  ", " ")  
                                            for x in l1.readlines()]
                
                # 2-character and 3-character alphabets are not \w (words) in re, switching to string.punctuation
                eng_words, tam_sentences = set(), []
                for sentence in l2.readlines():
                    # some english words show up in tamil dataset (lower case)
                    line = re.sub('\d+', ' %s ' % self.reserved_tokens[self.NUM_IDX], sentence.lower()) # use NUM reserved token
                    line = line.translate(translator) # remove punctuations
                    line = re.sub("\s+", " ", line) # correct for number of spaces
                    
                    p = re.compile("([a-z]+)\s|([a-z]+)")
                    search_results = p.search(line)
                    if not search_results is None:
                        eng_tokens = [x for x in search_results.groups() if not x is None]
                        eng_words.update(eng_tokens)

                    line = re.sub("[a-z]+\s|[a-z]+$", "%s " % self.reserved_tokens[self.ENG_IDX], line) # use ENG reserved token
                    tam_sentences.append(line.strip())

        for eng, tam in zip(eng_sentences, tam_sentences):
            text_pairs.append((eng, tam))
        
        random.shuffle(text_pairs)
        return text_pairs, eng_words
    
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
    
    def create_token_sentences_for_word2vec(self, eng_words):
        
        num_token_sentences = 500

        if len(eng_words) < num_token_sentences:
            eng_words = list(np.tile(eng_words, num_token_sentences//len(a) + 1)[:num_token_sentences])

        self.reserved_token_sentences = []
        for idx in range(len(eng_words)):
            string="%s " % self.reserved_tokens[self.BOS_IDX]
            string += eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.PAD_IDX]) * np.random.randint(0,3)
            string += eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.NUM_IDX]) * np.random.randint(0,3)
            string += eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.UNK_IDX]) * np.random.randint(0,3)
            string += eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.NUM_IDX]) * np.random.randint(0,3)
            string += eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.UNK_IDX]) * np.random.randint(0,3)
            string += eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.UNK_IDX]) * np.random.randint(0,3)
            string += eng_words[idx] if np.random.randint(0,2) else ""
            string += "%s " % self.reserved_tokens[self.EOS_IDX]
            string += ("%s " % self.reserved_tokens[self.PAD_IDX]) * np.random.randint(0,3)
            string += ("%s " % self.reserved_tokens[self.PAD_IDX]) * np.random.randint(0,3)
            string += ("%s " % self.reserved_tokens[self.PAD_IDX]) * np.random.randint(0,3)
            string = string.strip()
            
            self.reserved_token_sentences.append((string, string.replace(eng_words[idx], self.reserved_tokens[self.ENG_IDX])))

    def create_vocabulary(self, sentences, language='en'):
        
        assert language in ['en', 'ta']

        for idx, sentence in enumerate(sentences):
            sentences[idx] = self.reserved_tokens[self.BOS_IDX] + ' ' + sentence + ' ' + self.reserved_tokens[self.EOS_IDX]
        
        if hasattr(self, "reserved_token_sentences"):
            if language == 'en':
                sentences.extend([x[0] for x in self.reserved_token_sentences])
            elif language == 'ta':
                sentences.extend([x[1] for x in self.reserved_token_sentences])
        
        # English
        # 149309 before tokenization ; 75210 after
        # 70765 tokens without symbols
        # 67016 tokens without numbers
        
        # nltk vs stanza: 66952 vs 66942 tokens

        # Tamil
        # 271651 tokens with English words
        # 264429 tokens without English words (ENG tag)
        
        vocab = set()
        word_counts = {}
        
        for idx, sentence in enumerate(sentences):
            if idx == len(sentences) - 500:
                vocab.update(self.reserved_tokens)
                break
            else:
                if language == 'en':
                    #tokens = nltk.word_tokenize(sentence)
                    doc = en_nlp(sentence)
                    if len(doc.sentences) > 1:
                        tokens = [x.text for x in doc.sentences[0].tokens]
                        for sent in doc.sentences[1:]:
                            tokens.extend([x.text for x in sent.tokens])
                    else:
                        tokens = [x.text for x in doc.sentences[0].tokens]

                elif language == 'ta':
                    # stanza gives tokens of single alphabets that don't make semantic sense and increases vocab size
                    """
                    # Because of data preprocessing and special character removal, stanza doesn't do much for tokenizing tamil
                    #TODO check model performance with and without special characters
                    doc = ta_nlp(sentence)
                    
                    # ASSUMPTION: All sentence pairs are single sentences with no fullstop
                    if len(doc.sentences) > 1:
                        tokens = [x.text for x in doc.sentences[0].tokens]
                        for sent in doc.sentences[1:]:
                            tokens.extend([x.text for x in sent.tokens])
                    else:
                        tokens = [x.text for x in doc.sentences[0].tokens]
                    """

                    tokens = sentence.split(' ')

                for token in tokens:
                    if token in vocab:
                        word_counts[token] += 1
                    else:
                        word_counts[token] = 1
                
                vocab.update(tokens)
                sentences[idx] = " ".join(tokens)
        
        if language == "en":
            assert len(word_counts) == len(vocab) - (len(self.reserved_tokens) - 3), \
                    "Vocab size: %d, Word Count dictionary size: %d" % (len(vocab), len(word_counts)) # BOS, EOS, NUM already part of sentences
        else:
            assert len(word_counts) == len(vocab) - (len(self.reserved_tokens) - 4), \
                    "Vocab size: %d, Word Count dictionary size: %d" % (len(vocab), len(word_counts)) # BOS, EOS, NUM, ENG already part of sentences

        return vocab, word_counts, sentences

#train_dataset = EnTamV2Dataset("train")
#val_dataset = EnTamV2Dataset("dev")
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

