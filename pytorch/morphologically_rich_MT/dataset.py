import string
from unidecode import unidecode
import codecs
import binascii

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
from typing import Iterable, List

import os
import random
import re
import numpy as np
from collections import OrderedDict

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

#from gensim.models.fasttext import load_facebook_model
from gensim.models import Word2Vec

class EnTamV2Dataset(Dataset):

    SRC_LANGUAGE = 'en'
    TGT_LANGUAGE = 'ta'

    # Using START and END tokens in source and target vocabularies to enforce better relationships between x and y
    reserved_tokens = ["UNK", "PAD", "START", "END", "NUM", "ENG"]
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, NUM_IDX, ENG_IDX = 0, 1, 2, 3, 4, 5
    num_token_sentences = 500
  
    # Pretrained word2vec models take too long to load (many hours) - training my own
    #en_wv = load_facebook_model('dataset/monolingual/cc.en.300.bin_fasttext.bin')
    #ta_wv = load_facebook_model('dataset/monolingual/cc.ta.300.bin_fasttext.gz')

    def __init__(self, split, symbols=False):
        
        tokenized_dirname = "tokenized"
        if not os.path.exists(self.get_dataset_filename(split, "en", tokenized_dirname)) \
                or not os.path.exists(self.get_dataset_filename(split, "ta", tokenized_dirname)):
            
            self.bilingual_pairs, eng_words = self.get_sentence_pairs(split, symbols=symbols)
            
            if split == "train":
                eng_words = list(eng_words)
                self.create_token_sentences_for_word2vec(eng_words)

            self.eng_vocabulary, self.eng_word_counts, tokenized_eng_sentences = self.create_vocabulary([
                                                                                    x[0] for x in self.bilingual_pairs], language="en")
            self.tam_vocabulary, self.tam_word_counts, tokenized_tam_sentences = self.create_vocabulary([
                                                                                    x[1] for x in self.bilingual_pairs], language="ta")

            print ("Most Frequent 1000 English tokens:", sorted(self.eng_word_counts, key=lambda y: self.eng_word_counts[y], reverse=True)[:1000])
            print ("Most Frequent 1000 Tamil tokens:", sorted(self.tam_word_counts, key=lambda y: self.tam_word_counts[y], reverse=True)[:1000])

            # save tokenized sentences for faster loading

            with open(self.get_dataset_filename(split, "en", tokenized_dirname), 'w') as f:
                for line in tokenized_eng_sentences:
                    f.write("%s\n" % line)
            with open(self.get_dataset_filename(split, "ta", tokenized_dirname), 'w') as f:
                for line in tokenized_tam_sentences:
                    f.write("%s\n" % line)
            with open(self.get_dataset_filename(split, "en", tokenized_dirname, substr="vocab"), 'w') as f:
                for word in self.eng_vocabulary:
                    f.write("%s\n" % word)
            with open(self.get_dataset_filename(split, "ta", tokenized_dirname, substr="vocab"), 'w') as f:
                for word in self.tam_vocabulary:
                    f.write("%s\n" % word)
            self.bilingual_pairs = list(zip(tokenized_eng_sentences, tokenized_tam_sentences))
        
        else:
            
            with open(self.get_dataset_filename(split, "en", tokenized_dirname), 'r') as f:
                tokenized_eng_sentences = [x.strip() for x in f.readlines()]
            with open(self.get_dataset_filename(split, "ta", tokenized_dirname), 'r') as f:
                tokenized_tam_sentences = [x.strip() for x in f.readlines()]
            with open(self.get_dataset_filename(split, "en", tokenized_dirname, substr="vocab"), 'r') as f:
                self.eng_vocabulary = [x.strip() for x in f.readlines()]
            with open(self.get_dataset_filename(split, "ta", tokenized_dirname, substr="vocab"), 'r') as f:
                self.tam_vocabulary = [x.strip() for x in f.readlines()]
            self.bilingual_pairs = list(zip(tokenized_eng_sentences, tokenized_tam_sentences))
        
        assert "DEBUG" not in self.eng_vocabulary, "Debug token found in final train dataset"

        print ("English vocabulary size for %s set: %d" % (split, len(self.eng_vocabulary)))
        print ("Tamil vocabulary size for %s set: %d" % (split, len(self.tam_vocabulary)))
        
        print ("Using %s set with %d sentence pairs" % (split, len(self.bilingual_pairs)))

        if not os.path.exists('utils/Correlation.png') and split == "train":
            visualize_dataset_for_bucketing_stats(self.bilingual_pairs)
    
        # word2vec training
        self.train_word2vec_model_on_monolingual_and_mt_corpus(symbols)

    def train_word2vec_model_on_monolingual_and_mt_corpus(self, symbols):

        with open(self.get_dataset_filename("train", "en", subdir="word2vec", substr="word2vec"), 'r') as f:
            eng_word2vec = [x.strip() for x in f.readlines()]

        with open(self.get_dataset_filename("train", "ta", subdir="word2vec", substr="word2vec"), 'r') as f:
            tam_word2vec = [x.strip() for x in f.readlines()]
        
        # DEBUG
        en_word2vec = eng_word2vec[:1000]
        ta_word2vec = tam_word2vec[:1000]
       
        print ("Preprocessing word2vec datasets for English and Tamil")
        word2vec_sentences, word2vec_eng_words = self.get_sentence_pairs("train", symbols=symbols, dataset=[eng_word2vec, tam_word2vec])
        en_word2vec, ta_word2vec = word2vec_sentences
        
        print ("Tokenizing word2vec English and Tamil monolingual corpora")
        _,_, en_word2vec = self.create_vocabulary(en_word2vec, language="en")
        _,_, ta_word2vec = self.create_vocabulary(ta_word2vec, language="ta")
        
        en_word2vec = [x.split(' ') for x in en_word2vec]
        ta_word2vec = [x.split(' ') for x in ta_word2vec]
        
        print ("Training word2vec vocabulary for English")
        self.en_wv = Word2Vec(sentences=en_word2vec, vector_size=100, window=5, min_count=2, workers=4)
        self.en_wv.save("dataset/word2vec/word2vec.en.model")

        print ("Training word2vec vocabulary for Tamil")
        self.ta_wv = Word2Vec(sentences=ta_word2vec, vector_size=100, window=5, min_count=2, workers=4)
        self.ta_wv.save("dataset/word2vec/word2vec.ta.model")

        print ("car, minivan", self.en_wv.wv.most_similar(positive=['car', 'minivan'], topn=5))
        print ("இரண்ட", self.ta_wv.wv.most_similar(positive=['இரண்ட'], topn=5))
    
    def __len__(self):
        return len(self.bilingual_pairs)

    def get_dataset_filename(self, split, lang, subdir=None, substr=""): 
        assert split in ['train', 'dev', 'test', ''] and lang in ['en', 'ta', ''] # Using '' to get dirname because dataset was defined first here!
        assert substr in ["", "vocab", "word2vec"]
        if not subdir is None:
            if substr not in ["vocab", "word2vec"]:
                directory = os.path.join("dataset", subdir, "%s.bcn" % "corpus")
            else:
                directory = os.path.join("dataset", subdir, "%s.bcn" % substr)
        else:
            if substr not in ["vocab", "word2vec"]:
                directory = os.path.join("dataset", "%s.bcn" % "corpus")
            else:
                raise AssertionError

        full_path = "%s.%s.%s" % (directory, split, lang)
        
        save_dir = os.path.dirname(full_path)
        
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        
        return full_path

    def get_sentence_pairs(self, split, symbols=False, dataset=None):
        # use symbols flag to keep/remove punctuation

        text_pairs = []
        translator = str.maketrans('', '', string.punctuation)
        
        unnecessary_symbols = ["‘", "¦", "¡", "¬", '“', '”', "’", '\u200c'] # negation symbol might not be in EnTamV2
        # Exclamation mark between words in train set
        
        if symbols:
            symbol_replacements = {unnecessary_symbols[0]: "'", unnecessary_symbols[4]: '"', unnecessary_symbols[5]: "\"", unnecessary_symbols[6]: "'"}
        else:
            symbol_replacements = {}
        
        if not dataset is None:
            eng_sentences = dataset[0]
        else:
            with open(self.get_dataset_filename(split, self.SRC_LANGUAGE), 'r') as l1:
                eng_sentences = [re.sub(
                                    '\d+', ' %s ' % self.reserved_tokens[self.NUM_IDX], x.lower() # replace all numbers with [NUM] token
                                    ).strip().replace("  ", " ")
                                            for x in l1.readlines()]
            
        for idx, sentence in enumerate(eng_sentences):
            
            for sym in symbol_replacements:
                eng_sentences[idx] = eng_sentences[idx].replace(sym, " " + symbol_replacements[sym] + " ")
            
            if not symbols:
                eng_sentences[idx] = re.sub(r'[^\w\s*]|_',r' ', sentence)
            else:
                # couldn't use re here, not sure why
                for ch in string.punctuation:
                    eng_sentences[idx] = eng_sentences[idx].replace(ch, " "+ch+" ")

            for sym_idx, sym in enumerate(unnecessary_symbols):
                #return_in_bilingual_corpus = eng_sentences[idx]
                if not symbols or (symbols and not sym in symbol_replacements.keys()):
                    eng_sentences[idx] = eng_sentences[idx].replace(sym, "")

            eng_sentences[idx] = re.sub("\s+", " ", eng_sentences[idx]) # correct for number of spaces
            
            # take care of fractions: only 1/2 found - replacing with English word
            #eng_sentences[idx] = re.sub("\d+\s*/\s*\d+", "%s / %s" % (self.reserved_tokens[self.NUM_IDX], self.reserved_tokens[self.NUM_IDX]), eng_sentences[idx])
            
            # manual corrections
            eng_sentences[idx] = eng_sentences[idx].replace("naa-ve", "naive")
            eng_sentences[idx] = re.sub(r"j ' (\w)", r"i \1", eng_sentences[idx])
            eng_sentences[idx] = eng_sentences[idx].replace(". . .", "...")
        
        if not dataset is None:
            tam_sentences_file = dataset[1]
        else:
            with open(self.get_dataset_filename(split, self.TGT_LANGUAGE), 'r') as l2:
                # 2-character and 3-character alphabets are not \w (words) in re, switching to string.punctuation

                tam_sentences_file = list(l2.readlines())
            
        eng_words, tam_sentences = set(), []
        for idx, sentence in enumerate(tam_sentences_file):
        
            # some english words show up in tamil dataset (lower case)
            line = re.sub('\d+', ' %s ' % self.reserved_tokens[self.NUM_IDX], sentence.lower()) # use NUM reserved token

            if not symbols:
                line = line.translate(translator) # remove punctuations
            else:
                # couldn't use re here, not sure why
                for ch in string.punctuation:
                    line = line.replace(ch, " "+ch+" ")
                
                for sym in symbol_replacements:
                    line = line.replace(sym, " "+symbol_replacements[sym]+" ")

            for sym in unnecessary_symbols:
                if not symbols or (symbols and not sym in symbol_replacements.keys()):
                    line = line.replace(sym, "") 
            
            line = re.sub("\s+", " ", line) # correct for number of spaces
            
            if dataset is None:
                p = re.compile("([a-z]+)\s|([a-z]+)")
                search_results = p.search(line)
                if not search_results is None:
                    eng_tokens = [x for x in search_results.groups() if not x is None]
                    eng_words.update(eng_tokens)

                    with open(self.get_dataset_filename("train", "en", subdir="tamil_eng_vocab_untokenized"), 'a') as f:
                        f.write("%s\n" % (eng_sentences[idx]))
                    with open(self.get_dataset_filename("train", "ta", subdir="tamil_eng_vocab_untokenized"), 'a') as f:
                        f.write("%s\n" % (sentence))

            line = re.sub("[a-z]+\s*$", "%s " % self.reserved_tokens[self.ENG_IDX], line) # use ENG reserved token
            
            line.replace(". . .", "...")
            tam_sentences.append(line.strip())
        
        if dataset is None:
            for eng, tam in zip(eng_sentences, tam_sentences):
                text_pairs.append((eng, tam))
        
            random.shuffle(text_pairs)
        
            return text_pairs, eng_words
        
        else: #word2vec
            
            return [eng_sentences, tam_sentences], eng_words

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
        
        # DEBUG
        # tamil sentence has no english words for transfer to english vocabulary
        if len(eng_words) == 0:
            eng_words = ["DEBUG"]

        # instantiate for train set only
        self.eng_words = eng_words

        if len(eng_words) < self.num_token_sentences:
            eng_words = list(np.tile(eng_words, self.num_token_sentences//len(eng_words) + 1)[:self.num_token_sentences])

        self.reserved_token_sentences = []
        for idx in range(len(eng_words)):
            string="%s " % self.reserved_tokens[self.BOS_IDX]
            string += "%s " % eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.PAD_IDX]) * np.random.randint(0,3)
            string += "%s " % eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.NUM_IDX]) * np.random.randint(0,3)
            string += "%s " % eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.UNK_IDX]) * np.random.randint(0,3)
            string += "%s " % eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.NUM_IDX]) * np.random.randint(0,3)
            string += "%s " % eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.UNK_IDX]) * np.random.randint(0,3)
            string += "%s " % eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.UNK_IDX]) * np.random.randint(0,3)
            string += "%s " % eng_words[idx] if np.random.randint(0,2) else ""
            string += "%s " % self.reserved_tokens[self.EOS_IDX]
            string += ("%s " % self.reserved_tokens[self.PAD_IDX]) * np.random.randint(0,3)
            string += ("%s " % self.reserved_tokens[self.PAD_IDX]) * np.random.randint(0,3)
            string += ("%s " % self.reserved_tokens[self.PAD_IDX]) * np.random.randint(0,3)
            string = string.strip()
            
            src_string = string.replace(self.reserved_tokens[self.UNK_IDX], eng_words[idx])
            trg_string = string.replace(eng_words[idx], self.reserved_tokens[self.ENG_IDX])
            self.reserved_token_sentences.append((src_string, trg_string))

    def create_vocabulary(self, sentences, language='en'):
        
        assert language in ['en', 'ta']

        for idx in range(len(sentences)):
            if language == 'en':
                sentences[idx] = unidecode(sentences[idx]) # remove accents from english sentences
                # Refer FAQs here: https://pypi.org/project/Unidecode/
                sentences[idx] = sentences[idx].replace("a<<", "e") # umlaut letter
                sentences[idx] = sentences[idx].replace("a 1/4", "u") # u with diaeresis
                sentences[idx] = sentences[idx].replace("a3", "o") # ó: a3 --> o
                
                sentences[idx] = sentences[idx].replace("a(r)", "i") # î: a(r) --> i
                sentences[idx] = sentences[idx].replace("a-", "i") # ï: a- --> i [dataset seems to also use ocr: ïடக்கக்கூடியதுதான்  --> i(da)kka for padi[kka]]
                sentences[idx] = sentences[idx].replace("a$?", "a") # ä: a$? --> a
                sentences[idx] = sentences[idx].replace("a'", "o") # ô: a' --> o
                sentences[idx] = sentences[idx].replace("d1", "e") # econostrum - single token
                sentences[idx] = sentences[idx].replace("a+-", "n") # ñ: a+- --> n
                sentences[idx] = sentences[idx].replace("a1", "u") # ù: a1 --> u
            
                # manual change
                num_and_a_half = lambda x: "%s%s" % (self.reserved_tokens[self.NUM_IDX], x) # NUM a half --> NUM and a half
                sentences[idx] = sentences[idx].replace(num_and_a_half(" a 1/2"), num_and_a_half(" and a half"))
            
            sentences[idx] = self.reserved_tokens[self.BOS_IDX] + ' ' + sentences[idx] + ' ' + self.reserved_tokens[self.EOS_IDX]
            sentences[idx] = re.sub('\s+', ' ', sentences[idx])

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
        
        virama_introduction_chars = {"ங": "ங்"}

        if hasattr(self, "eng_tokens"):
            vocab.update(self.eng_tokens)

        for idx, sentence in enumerate(sentences):
            if idx == len(sentences) - self.num_token_sentences and hasattr(self, 'reserved_token_sentences'):
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
                    # Because of data preprocessing and special character removal, stanza doesn't do much for tokenizing tamil

                    tokens = sentence.split(' ')
                    
                    # sentence 32579: 
                    for token_index, token in enumerate(tokens):

                        if not token in string.punctuation:
                            
                            token_languages = self.get_entam_sequence(token)
                            token_replacement = self.tokenize_entam_combinations(token_languages, token)
                            
                            if token_replacement[0] == token:
                                continue
                            else:
                                new_sentence_tokens = tokens[:token_index] + token_replacement + tokens[token_index+1:]
                                sentences[idx] = " ".join(new_sentence_tokens)

                for token_idx, token in enumerate(tokens):
                    
                    # use stress character (virama from wikipedia) to end tokens that need them
                    if language == "ta" and token[-1] in virama_introduction_chars.keys():
                        token = token[:-1] + virama_introduction_chars[token[-1]]

                    if token in vocab:
                        word_counts[token] += 1
                    else:
                        word_counts[token] = 1
                
                vocab.update(tokens)
                sentences[idx] = " ".join(tokens)
                
        if hasattr(self, "eng_vocab"):
            if language == "en":
                tokens_in_eng_vocabulary = 4 # only UNK and ENG don't belong to en vocabulary
                assert len(word_counts) == len(vocab) - (len(self.reserved_tokens) - tokens_in_eng_vocabulary), \
                        "sentence %d: Vocab size: %d, Word Count dictionary size: %d" % (idx, len(vocab), len(word_counts)) # BOS, EOS, NUM, PAD already part of sentences
            else:
                assert len(word_counts) == len(vocab), \
                        "sentence %d: Vocab size: %d, Word Count dictionary size: %d" % (idx, len(vocab), len(word_counts)) # BOS, EOS, NUM, PAD, ENG already part of sentences

        return vocab, word_counts, sentences

    def tokenize_entam_combinations(self, token_languages, token):

        tokens_split, tamil_part = [], ""
        keys = list(token_languages.keys())
        for idx, key in enumerate(reversed(keys[:-1])):
            lang = "en" if "en" in key else "ta"
            start_of_lang_block = token_languages[key]
            end_of_lang_block = token_languages[keys[len(keys)-1 - idx]]
            
            if lang == "en":
                if end_of_lang_block - start_of_lang_block >= 3:
                    if tamil_part == "":
                        tokens_split.append(self.reserved_tokens[self.ENG_IDX])
                    else:
                        tokens_split.extend([tamil_part, self.reserved_tokens[self.ENG_IDX]])
                        tamil_part = ""
            else:
                tamil_part = token[start_of_lang_block:end_of_lang_block] + tamil_part
        
        if tamil_part != "":
            tokens_split.append(tamil_part)
        else:
            # no tamil characters means <=2 character english token
            tokens_split.append(self.reserved_tokens[self.ENG_IDX])

        tokens_split = list(reversed(tokens_split))

        return tokens_split
    
    def get_entam_sequence(self, token):
        
        if not hasattr(self, "tamil_characters_hex"):
            self.tamil_characters_hex = self.return_tamil_unicode_isalnum()
        
        sequence = OrderedDict()
        num_eng, num_tam = 0, 0
        get_count = lambda lang: str(num_eng) if lang=='en' else str(num_tam)

        unicode_hex = "".join("{:02x}".format(ord(x)) for x in token[0])
        if unicode_hex in self.tamil_characters_hex:
            lang = 'ta'
            num_tam += 1
        else:
            lang = 'en'
            num_eng += 1

        sequence[lang+"0"] = 0

        for idx, character in enumerate(list(token)[1:]):

            unicode_hex = "".join("{:02x}".format(ord(x)) for x in character)
            
            if unicode_hex in self.tamil_characters_hex:
                if lang == 'en':
                    lang = 'ta'
                    sequence[lang+get_count(lang)] = idx + 1
                    num_tam += 1
            else:
                if lang == 'ta':
                    lang = 'en'
                    sequence[lang+get_count(lang)] = idx + 1
                    num_eng += 1

        sequence[lang+get_count(lang)] = len(token)
        return sequence    
    
    def return_unicode_hex_within_range(self, start, num_chars):
        assert isinstance(start, str) and isinstance(num_chars, int)
        start_integer = int(start, 16)
        return ["".join("{:02x}".format(start_integer+x)).lower() for x in range(num_chars)]

    def return_tamil_unicode_isalnum(self):
        ayudha_yezhuthu_stressing_connector = self.return_unicode_hex_within_range("b82", 2)
        a_to_ooo = self.return_unicode_hex_within_range("b85", 6)
        ye_to_i = self.return_unicode_hex_within_range("b8e", 3)
        o_O_ou_ka = self.return_unicode_hex_within_range("b92", 4)
        nga_sa = self.return_unicode_hex_within_range("b99", 2)
        ja = self.return_unicode_hex_within_range("b9c", 1)
        nya_ta = self.return_unicode_hex_within_range("b9e", 2)
        Na_tha = self.return_unicode_hex_within_range("ba3", 2)
        na_na_pa = self.return_unicode_hex_within_range("ba8", 3)
        ma_yararavazhaLa_sa_ssa_sha_ha = self.return_unicode_hex_within_range("bae", 12)
        aa_e_ee_oo_ooo_connectors = self.return_unicode_hex_within_range("bbe", 5)
        a_aay_ai_connectors = self.return_unicode_hex_within_range("bc6", 3)
        o_oo_ou_stressing_connectors = self.return_unicode_hex_within_range("bca", 4)
        ou = self.return_unicode_hex_within_range("bd0", 1)
        ou_length_connector = self.return_unicode_hex_within_range("bd7", 1)
        numbers_and_misc_signs = self.return_unicode_hex_within_range("be6", 21)
        
        all_chars = ayudha_yezhuthu_stressing_connector + a_to_ooo + ye_to_i + o_O_ou_ka + nga_sa
        all_chars += ja + nya_ta + Na_tha + na_na_pa + ma_yararavazhaLa_sa_ssa_sha_ha
        all_chars += aa_e_ee_oo_ooo_connectors + a_aay_ai_connectors + o_oo_ou_stressing_connectors 
        all_chars += ou + ou_length_connector + numbers_and_misc_signs

        return all_chars

    def get_tamil_special_characters(self, sentence, idx):

        if not hasattr(self, "tamil_characters_hex"):
            self.tamil_characters_hex = self.return_tamil_unicode_isalnum()
        
        if sentence in self.reserved_tokens:
            return [], "", False

        spl_chars, tamil_token, prefix = [], "", False
        for unicode_2_or_3 in sentence:
            # token level special character search doesn't need to check for space
            #if unicode_2_or_3 == ' ':
            #    continue

            unicode_hex = "".join("{:02x}".format(ord(x)) for x in unicode_2_or_3)
            if not unicode_hex in self.tamil_characters_hex:
                if len(spl_chars) == 0:
                    prefix = True
                if not unicode_2_or_3 in string.punctuation:
                    spl_chars.append(unicode_2_or_3)
            else:
                tamil_token += unicode_2_or_3
        
        assert len(spl_chars) + len(tamil_token) == len(sentence), \
                "sentence %d: Complicated English-Tamil combo word: %s (%d), spl chars: %s (%d), tamil: %s (%d)" % (
                        idx, sentence, len(sentence), spl_chars, len(spl_chars), tamil_token, len(tamil_token))

        return spl_chars, tamil_token, prefix

#train_dataset = EnTamV2Dataset("train")
#val_dataset = EnTamV2Dataset("dev")
#test_dataset = EnTamV2Dataset("test")

#train_dataset = EnTamV2Dataset("train", symbols=True)
val_dataset = EnTamV2Dataset("dev", symbols=True)
#test_dataset = EnTamV2Dataset("test", symbols=True)

exit()

# word2vec choices: 
# 1. one large model for both english and tamil allows english words to stay in tamil vocabulary 
#    to learn richer source and target related embeddings
# 2. two separate models allows ENG token in tamil vocabulary and closer word vectorization matches 
#    in the specific language datasets -- I didn't want english in tamil dataset
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

