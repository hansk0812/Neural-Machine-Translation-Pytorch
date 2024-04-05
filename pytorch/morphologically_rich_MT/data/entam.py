from logger import Logger
from preprocess import Preprocess
from bucket import Bucketing
from unicode_map import UnicodeMap

from cache import Cache
from vocabulary import Vocabulary
from word_embedding import WordEmbedding

from utils import get_sentences_from_file

import re
import string

from torch.utils.data import Dataset

from gensim.models import Word2Vec

class EnTam(Dataset, Logger):

    VERBOSE = True
    
    tamil_hex_ranges = [("b82", 2), ("b85", 6), ("b8e", 3), 
                        ("b92", 4), ("b99", 2), ("b9c", 1),
                        ("b9e", 2), ("ba3", 2), ("ba8", 3),
                        ("bae", 12), ("bbe", 5), ("bc6", 3),
                        ("bca", 4), ("bd0", 1), ("bd7", 1),
                        ("be6", 21)]

    reserved_tokens = ["UNK", "PAD", "ENG", "NUM", "START", "END"]

    def __init__(self, l1_fpath, l2_fpath, start_stop=True, verbose=True, cache_id=0,
                 buckets=[[5,5], [8,8], [12,12], [15,15], [18,18], [21,21], [24,24], [30,30], [40,40], [50,50], [80,80]],
                 bucketing_language_sort = "l2", max_vocab_size=150000):
        
        Logger.__init__(self, verbose)

        assert bucketing_language_sort in ["l1", "l2"]
        
        self.start_stop_tokens = start_stop

        l1_sentences, l2_sentences = get_sentences_from_file(l1_fpath, l2_fpath)

        #...........................................................................................................................................        
        # Preprocessing + Caching START
        #...........................................................................................................................................        

        entam_cache = Cache("cache", cache_id=cache_id)
        if not entam_cache.is_file("tokenized.en") or not entam_cache.is_file("tokenized.ta"):

            unnecessary_symbols = ["¦", "¡", "¬", '\u200c']
            symbol_replacements = {"‘": "'", '“': '"', '”': "\"", "’": "'"}

            # Preprocess
            self.preprocess = Preprocess(l1_sentences, l2_sentences, verbose=self.VERBOSE)
            self.preprocess.remove_symbols(list(string.punctuation) + unnecessary_symbols, list(string.punctuation) + unnecessary_symbols, symbol_replacements)
            self.preprocess.lower_case_english("l1")
            self.preprocess.reserved_token_num(self.reserved_tokens[3])
            self.preprocess.l1_sentences = self.preprocess.unidecode_english(self.preprocess.l1_sentences)
            #self.preprocess.l2_sentences = self.preprocess.unidecode_english(self.preprocess.l2_sentences) #transliteration
        
            self.preprocess.tokenize_english()

            removable_indices = []
            for idx in range(len(self.preprocess.l1_sentences)):
                if self.preprocess.l1_sentences[idx].strip() == "" or \
                        self.preprocess.l2_sentences[idx].strip() == "":
                    removable_indices.append(idx)
            for r in reversed(removable_indices):
                del self.preprocess.l1_sentences[r]
                del self.preprocess.l2_sentences[r]

            # Unicode code block restrictions
            tamil_map = UnicodeMap(language="Tamil", hex_ranges=self.tamil_hex_ranges, verbose=self.VERBOSE)
            for l2_idx in range(len(self.preprocess.l2_sentences)):
                sentence_tokens = []
                for token in self.preprocess.l2_sentences[l2_idx].split(' '):
                    sentence_tokens.extend(tamil_map.tokenize(token, self.reserved_tokens, eng_token="ENG"))
                self.preprocess.l2_sentences[l2_idx] = ' '.join(sentence_tokens)
        
            entam_cache.variable_to_file(self.preprocess.l1_sentences, "tokenized.en")
            entam_cache.variable_to_file(self.preprocess.l2_sentences, "tokenized.ta")
        
        else: # use cached preprocessed sentences
            
            l1_sentences = entam_cache.file_to_variable("tokenized.en")
            l2_sentences = entam_cache.file_to_variable("tokenized.ta")
            
            self.preprocess = Preprocess(l1_sentences, l2_sentences, verbose=self.VERBOSE)
        
        #...........................................................................................................................................        
        # Preprocessing + Caching END
        #...........................................................................................................................................        
        
        #...........................................................................................................................................        
        # START and STOP tokens for attention mechanism START
        #...........................................................................................................................................        

        if start_stop:
            for idx in range(len(self.preprocess.l1_sentences)):
                if self.start_stop_tokens:
                    self.preprocess.l1_sentences[idx] = self.reserved_tokens[4] + ' ' + \
                                                        self.preprocess.l1_sentences[idx] + \
                                                        ' ' + self.reserved_tokens[5]
                    self.preprocess.l2_sentences[idx] = self.reserved_tokens[4] + ' ' + \
                                                        self.preprocess.l2_sentences[idx] + \
                                                        ' ' + self.reserved_tokens[5]

                self.preprocess.l1_sentences[idx] = re.sub('\s+', ' ', self.preprocess.l1_sentences[idx])
                self.preprocess.l2_sentences[idx] = re.sub('\s+', ' ', self.preprocess.l2_sentences[idx])
        
        #...........................................................................................................................................        
        # START and STOP tokens for attention mechanism END
        #...........................................................................................................................................        
        
        #...........................................................................................................................................        
        # Bucketing before Word Embeddings and vocabulary START
        #...........................................................................................................................................        
        
        if not entam_cache.is_file("bucketed.en") or entam_cache.is_file("bucketed.ta"):
            bilingual_pairs = [[self.preprocess.l1_sentences[idx], self.preprocess.l2_sentences[idx]] \
                                for idx in range(len(self.preprocess.l1_sentences))]
            bilingual_pairs = sorted(bilingual_pairs, key=lambda x: len(x[bucketing_language_sort == "l2"].split(' ')))

            bucketer = Bucketing(bilingual_pairs, buckets=buckets, sort_order="l2", verbose=verbose)
        
            self.preprocess.l1_sentences = [x[0] for x in bucketer.bilingual_pairs]
            self.preprocess.l2_sentences = [x[1] for x in bucketer.bilingual_pairs]

            entam_cache.variable_to_file(self.preprocess.l1_sentences, "bucketed.en")
            entam_cache.variable_to_file(self.preprocess.l2_sentences, "bucketed.ta")
        else:
            self.preprocess.l1_sentences = entam_cache.file_to_variable("bucketed.en")
            self.preprocess.l2_sentences = entam_cache.file_to_variable("bucketed.ta")

        #...........................................................................................................................................        
        # Bucketing before Word Embeddings and vocabulary END
        #...........................................................................................................................................        
        
        #...........................................................................................................................................        
        # train set vocabulary START ( #TODO Word2vec monolingual corpus addition )
        #...........................................................................................................................................        
        
        if not entam_cache.is_file("vocabulary.en") or not entam_cache.is_file("vocabulary.ta"):
            self.l1_vocab = Vocabulary(self.preprocess.l1_sentences, self.reserved_tokens, language="English", verbose=verbose)
            self.l2_vocab = Vocabulary(self.preprocess.l2_sentences, self.reserved_tokens, language="Tamil", verbose=verbose)

            self.preprocess.l1_sentences = self.l1_vocab.restrict_vocabulary(max_vocab_size)
            self.preprocess.l2_sentences = self.l2_vocab.restrict_vocabulary(max_vocab_size)

            entam_cache.variable_to_file(self.l1_vocab.sorted_tokens, "vocabulary.en")
            entam_cache.variable_to_file(self.l2_vocab.sorted_tokens, "vocabulary.ta")
            entam_cache.variable_to_file(self.l1_vocab.sentences, "train_ready.en")
            entam_cache.variable_to_file(self.l2_vocab.sentences, "train_ready.ta")
        else:
            self.l1_vocab = Vocabulary(self.preprocess.l1_sentences, self.reserved_tokens, language="English", verbose=verbose, count=False)
            self.l1_vocab.sorted_tokens = entam_cache.file_to_variable("vocabulary.en")
            self.l1_vocab.token_indices = {token: self.l1_vocab.sorted_tokens.index(token) for token in self.l1_vocab.sorted_tokens}
            self.l1_vocab.sentences = entam_cache.file_to_variable("train_ready.en")
            
            self.l2_vocab = Vocabulary(self.preprocess.l2_sentences, self.reserved_tokens, language="English", verbose=verbose, count=False)
            self.l2_vocab.sorted_tokens = entam_cache.file_to_variable("vocabulary.ta")
            self.l2_vocab.token_indices = {token: self.l2_vocab.sorted_tokens.index(token) for token in self.l2_vocab.sorted_tokens}
            self.l2_vocab.sentences = entam_cache.file_to_variable("train_ready.ta")
        
        self.print ('Most frequent tokens in vocabularies')
        self.print (self.l1_vocab.sorted_tokens[:100])
        self.print (self.l2_vocab.sorted_tokens[:500])

        #...........................................................................................................................................        
        # train set vocabulary END ( #TODO Word2vec monolingual corpus addition )
        #...........................................................................................................................................        
        
        #...........................................................................................................................................        
        # Word2vec
        #...........................................................................................................................................        
        
        if not entam_cache.is_file("word2vec.en") or not entam_cache.is_file("word2vec.ta"):

            self.print ("Preprocessing word2vec datasets for English and Tamil")
            
            self.print ("Training word2vec vocabulary for English")
            self.l1_wv = Word2Vec(sentences=[x.split(' ') for x in self.preprocess.l1_sentences], vector_size=100, window=5, min_count=1, workers=4)
            #self.l1_wv.build_vocab(self.preprocess.l1_sentences) # placeholder for larger monolingual corpus #TODO
            #self.l1_wv.train(self.preprocess.l1_sentences, total_examples=len(self.preprocess.l1_sentences), epochs=20)
            self.l1_wv.save(entam_cache.get_path("word2vec.en"))

            self.print ("Training word2vec vocabulary for Tamil")
            self.l2_wv = Word2Vec(sentences=[x.split(' ') for x in self.preprocess.l2_sentences], vector_size=100, window=5, min_count=1, workers=4)
            #self.l2_wv.build_vocab(self.preprocess.l2_sentences) # placeholder for larger monolingual corpus #TODO
            #self.l2_wv.train(self.preprocess.l2_sentences, total_examples=len(self.preprocess.l2_sentences), epochs=20)
            self.l2_wv.save(entam_cache.get_path("word2vec.ta"))
        
        else:

            self.l1_wv = Word2Vec.load(entam_cache.get_path('word2vec.en'))
            self.l2_wv = Word2Vec.load(entam_cache.get_path('word2vec.ta'))
        
        self.print("Word2vec models loaded")

        self.l1_embedding = [self.l1_wv.wv[token] for token in self.l1_vocab.sorted_tokens]
        self.l2_embedding = [self.l2_wv.wv[token] for token in self.l2_vocab.sorted_tokens]

        #...........................................................................................................................................        
        # Word2vec
        #...........................................................................................................................................        



if __name__ == "__main__":

    dataset = EnTam("../dataset/corpus.bcn.train.en", "../dataset/corpus.bcn.train.ta")

    idx = 5
    for l1, l2 in zip(dataset.preprocess.l1_sentences, dataset.preprocess.l2_sentences):
        print (l1)
        print ()
        print (l2)
        print ('.'*75)
        print ()
        if idx == 0:
            break
        idx -= 1
