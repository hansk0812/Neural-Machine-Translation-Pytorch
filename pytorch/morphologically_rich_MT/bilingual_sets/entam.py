from data.logger import Logger
from data.preprocess import Preprocess
from data.bucket import Bucketing
from data.unicode_map import UnicodeMap

from data.cache import Cache
from data.vocabulary import Vocabulary
from data.word_embedding import WordEmbedding

from data.utils import get_sentences_from_file, BucketingBatchSamplerReplace as BucketingBatchSampler
#BucketingBatchSampler

from data import reserved_tokens, tamil_hex_ranges

import re
import string

import numpy as np

from torch.utils.data import Dataset, DataLoader

from gensim.models import Word2Vec

class EnTam(Dataset, Logger):

    reserved_tokens = reserved_tokens
    tamil_hex_ranges = tamil_hex_ranges

    def __init__(self, l1_fpath, l2_fpath, start_stop=True, verbose=True, cache_id=0, vocabularies=None, word2vecs=None,
                 buckets=[[5,5], [8,8], [12,12], [15,15], [18,18], [21,21], [24,24], [30,30], [40,40], [50,50]],
                 bucketing_language_sort = "l2", max_vocab_size=150000, morphemes=False):
        
        Logger.__init__(self, verbose)

        self.morphemes = morphemes

        assert bucketing_language_sort in ["l1", "l2"]
        
        self.start_stop_tokens = start_stop

        l1_sentences, l2_sentences = get_sentences_from_file(l1_fpath, l2_fpath)
        
        # DATASET VARIABLE: l1_sentences, l2_sentences
        #...........................................................................................................................................        
        # Preprocessing + Caching START
        #........................................................................................................................................... 
        # DATASET VARIABLE: self.preprocess.l1_sentences, self.preprocess.l2_sentences

        tamil_map = UnicodeMap(language="Tamil", hex_ranges=self.tamil_hex_ranges, verbose=verbose)
        entam_cache = Cache("cache", cache_id=cache_id)
        if not entam_cache.is_file("tokenized.en") or not entam_cache.is_file("tokenized.ta"):

            unnecessary_symbols = ["¦", "¡", "¬", '\u200c']
            symbol_replacements = {"‘": "'", '“': '"', '”': "\"", "’": "'"}

            # Preprocess
            self.preprocess = Preprocess(l1_sentences, l2_sentences, verbose=verbose)
            self.preprocess.remove_symbols(list(string.punctuation) + unnecessary_symbols, list(string.punctuation) + unnecessary_symbols, symbol_replacements)
            self.preprocess.lower_case_english("l1")
            self.preprocess.lower_case_english("l2")
            self.preprocess.reserved_token_num(self.reserved_tokens[3])
            self.preprocess.l1_sentences = self.preprocess.unidecode_english(self.preprocess.l1_sentences)
            #self.preprocess.l2_sentences = self.preprocess.unidecode_english(self.preprocess.l2_sentences) #transliteration
        
            self.preprocess.tokenize_english()

            """
            removable_indices = []
            for idx in range(len(self.preprocess.l1_sentences)):
                if self.preprocess.l1_sentences[idx].strip() == "" or \
                        self.preprocess.l2_sentences[idx].strip() == "":
                    removable_indices.append(idx)
            for r in reversed(removable_indices):
                del self.preprocess.l1_sentences[r]
                del self.preprocess.l2_sentences[r]
            """

            # Unicode code block restrictions
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
            
            self.preprocess = Preprocess(l1_sentences, l2_sentences, verbose=verbose)
        
        # Ensure adding UNK tokens don't go through preprocessing
        for _ in range(50):
            unk_sentence = "%s %s %s" % (self.reserved_tokens[4], self.reserved_tokens[0], reserved_tokens[5])
            self.preprocess.l1_sentences.append(unk_sentence)
            self.preprocess.l2_sentences.append(unk_sentence)

        # DATASET VARIABLE: self.preprocess.l1_sentences, self.preprocess.l2_sentences
        #...........................................................................................................................................        
        # Preprocessing + Caching END
        #...........................................................................................................................................        
        
        #...........................................................................................................................................        
        # START and STOP tokens for attention mechanism START
        #...........................................................................................................................................        

        if start_stop:
            for idx in range(len(self.preprocess.l1_sentences)):
                self.preprocess.l1_sentences[idx] = self.reserved_tokens[4] + ' ' + \
                                                    self.preprocess.l1_sentences[idx] + \
                                                    ' ' + self.reserved_tokens[5]
                self.preprocess.l2_sentences[idx] = self.reserved_tokens[4] + ' ' + \
                                                    self.preprocess.l2_sentences[idx] + \
                                                    ' ' + self.reserved_tokens[5]

                self.preprocess.l1_sentences[idx] = re.sub('\s+', ' ', self.preprocess.l1_sentences[idx])
                self.preprocess.l2_sentences[idx] = re.sub('\s+', ' ', self.preprocess.l2_sentences[idx])
        
        # DATASET VARIABLE: self.preprocess.l1_sentences, self.preprocess.l2_sentences
        #...........................................................................................................................................        
        # START and STOP tokens for attention mechanism END
        #...........................................................................................................................................        
        # DATASET VARIABLE: self.preprocess.l1_sentences, self.preprocess.l2_sentences
        
        #...........................................................................................................................................        
        # Bucketing before Word Embeddings and vocabulary START
        #...........................................................................................................................................        
        # DATASET VARIABLE: l1_sentences, l2_sentences <--> bilingual_pairs - sorting
        # sort once before bucketing to find bucketing_indices for dataloader, sort after for PAD based sorting
        
        self.bilingual_pairs = [[self.preprocess.l1_sentences[idx], self.preprocess.l2_sentences[idx]] \
                            for idx in range(len(self.preprocess.l1_sentences))]

        # sort 1: before bucketing
        self.bilingual_pairs = sorted(self.bilingual_pairs, key=lambda x: len(x[bucketing_language_sort == "l2"].split(' ')))
        
        # sort 2: after bucketing
        self.bucketer = Bucketing(self.bilingual_pairs, buckets=buckets, sort_order="l2", verbose=verbose)
        
        self.bilingual_pairs = self.bucketer.bilingual_pairs

        # bilingual_pairs == l1_sentences, l2_sentences
        #...........................................................................................................................................        
        # Bucketing before Word Embeddings and vocabulary END
        #...........................................................................................................................................        
        
        # DATASET: bilingual_pairs --> per vocabulary sentences
        #...........................................................................................................................................        
        # train set vocabulary START ( #TODO Word2vec monolingual corpus addition )
        #...........................................................................................................................................        
        # DATASET: l1_sentences, l2_sentences; vocabulary indifferent to dataset;
        
        if vocabularies is None:

            if not entam_cache.is_file("vocabulary.en") or not entam_cache.is_file("vocabulary.ta") \
                    or not entam_cache.is_file("train_ready.en") or not entam_cache.is_file("train_ready.ta"):

                self.l1_vocab = Vocabulary([x[0] for x in self.bilingual_pairs], self.reserved_tokens[:1], language="English", new_vocab_size=max_vocab_size, verbose=verbose)
                self.l2_vocab = Vocabulary([x[1] for x in self.bilingual_pairs], self.reserved_tokens, language="Tamil",  new_vocab_size=max_vocab_size, verbose=verbose)
            
                #print ([(len(x[0]), len(x[1])) for x in self.bilingual_pairs])

                #l1_sentences = self.l1_vocab.restrict_vocabulary(max_vocab_size)
                #l2_sentences = self.l2_vocab.restrict_vocabulary(max_vocab_size)
                
                l1_sentences = self.l1_vocab.sentences
                l2_sentences = self.l2_vocab.sentences

                self.biligual_pairs = list(zip(l1_sentences, l2_sentences))
                del l1_sentences, l2_sentences

                entam_cache.variable_to_file(self.l1_vocab.sorted_tokens, "vocabulary.en")
                entam_cache.variable_to_file(self.l2_vocab.sorted_tokens, "vocabulary.ta")
                entam_cache.variable_to_file([x[0] for x in self.bilingual_pairs], "train_ready.en")
                entam_cache.variable_to_file([x[1] for x in self.bilingual_pairs], "train_ready.ta")
            else:
                self.l1_vocab = Vocabulary([x[0] for x in self.bilingual_pairs], self.reserved_tokens, language="English", 
                                            new_vocab_size=max_vocab_size, verbose=verbose, count=True)
                self.l1_vocab.sorted_tokens = entam_cache.file_to_variable("vocabulary.en")
                self.l1_vocab.token_indices = {token: self.l1_vocab.sorted_tokens.index(token) for token in self.l1_vocab.sorted_tokens}
                l1_sentences = entam_cache.file_to_variable("train_ready.en")
                
                self.l2_vocab = Vocabulary([x[1] for x in self.bilingual_pairs], self.reserved_tokens, language="English", 
                                            new_vocab_size=max_vocab_size, verbose=verbose, count=True)
                self.l2_vocab.sorted_tokens = entam_cache.file_to_variable("vocabulary.ta")
                self.l2_vocab.token_indices = {token: self.l2_vocab.sorted_tokens.index(token) for token in self.l2_vocab.sorted_tokens}
                l2_sentences = entam_cache.file_to_variable("train_ready.ta")
                
                self.biligual_pairs = list(zip(l1_sentences, l2_sentences))
                del l1_sentences, l2_sentences
        else:
            self.l1_vocab, self.l2_vocab = vocabularies

        self.print ('Most frequent tokens in vocabularies')
        self.print (self.l1_vocab.sorted_tokens[:100])
        self.print (self.l2_vocab.sorted_tokens[:500])

        #...........................................................................................................................................        
        # train set vocabulary END ( #TODO Word2vec monolingual corpus addition )
        #...........................................................................................................................................        
        
        #...........................................................................................................................................        
        # Word2vec
        #...........................................................................................................................................        
        # DATASET: bilingual_pairs

        if word2vecs is None:
            if not entam_cache.is_file("word2vec.en") or not entam_cache.is_file("word2vec.ta"):

                self.print ("Preprocessing word2vec datasets for English and Tamil")
                
                self.print ("Training word2vec vocabulary for English")
                self.l1_wv = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
                self.l1_wv.build_vocab([x[0].split(' ') for x in self.bilingual_pairs]) # placeholder for larger monolingual corpus #TODO
                self.l1_wv.train([x[0].split(' ') for x in self.bilingual_pairs], total_examples=len(self.bilingual_pairs), epochs=20)
                self.l1_wv.save(entam_cache.get_path("word2vec.en"))

                self.print ("Training word2vec vocabulary for Tamil")
                self.l2_wv = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
                self.l2_wv.build_vocab([x[1].split(' ') for x in self.bilingual_pairs]) # placeholder for larger monolingual corpus #TODO
                self.l2_wv.train([x[1].split(' ') for x in self.bilingual_pairs], total_examples=len(self.bilingual_pairs), epochs=20)
                self.l2_wv.save(entam_cache.get_path("word2vec.ta"))
            
            else:

                self.l1_wv = Word2Vec.load(entam_cache.get_path('word2vec.en'))
                self.l2_wv = Word2Vec.load(entam_cache.get_path('word2vec.ta'))
        
        else:
            self.l1_wv, self.l2_wv = word2vecs

        self.print("Word2vec models loaded")
        
        if word2vecs is None:
            self.l1_embedding = [self.l1_wv.wv[token] for token in self.l1_vocab.sorted_tokens]
            self.l2_embedding = [self.l2_wv.wv[token] for token in self.l2_vocab.sorted_tokens]
        
        #...........................................................................................................................................        
        # Word2vec
        #...........................................................................................................................................        
        # DATASET: bilingual_pairs --> bilingual_indices for training

        self.pad_idx = self.l1_vocab.sorted_tokens.index(self.reserved_tokens[1])

        for data_idx in range(len(self.bilingual_pairs)):
            en, ta = self.bilingual_pairs[data_idx]
            
            en_tokens, ta_tokens = [], []
            for token in en.split(' '):
                try:
                    index = self.l1_vocab.token_indices[token]
                    en_tokens.append(index)
                except Exception:
                    self.print ("English: %s --> UNK" % token)
                    en_tokens.append(self.l1_vocab.token_indices[self.reserved_tokens[0]])
            for token in ta.split(' '):
                try:
                    index = self.l2_vocab.token_indices[token]
                    ta_tokens.append(index)
                except Exception:
                    if not tamil_map.get_membership(token):
                        self.print ("Tamil: %s --> ENG" % token)
                        ta_tokens.append(self.l2_vocab.token_indices[self.reserved_tokens[2]])
                    else:
                        self.print ("Tamil: %s --> UNK" % token)
                        ta_tokens.append(self.l2_vocab.token_indices[self.reserved_tokens[0]])

            self.bilingual_pairs[data_idx] = [en_tokens, ta_tokens]

    def return_vocabularies(self):
        return self.l1_vocab, self.l2_vocab
    
    def return_word2vecs(self):
        return self.l1_wv, self.l2_wv

    def indices_to_words(self, indices, language):

        assert language in ["en", "ta"]
        
        if language == "en":
            ret = [self.l1_vocab.token_indices_reverse[k] for k in indices]
        else:
            ret = [self.l2_vocab.token_indices_reverse[k] for k in indices]

        return " ".join(ret)

    def __len__(self):
        return len(self.bilingual_pairs)

    def __getitem__(self, idx):
       
        """
        eng_sentence = self.preprocess.l1_sentences[idx]
        tam_sentence = self.preprocess.l2_sentences[idx]
        """

        E = self.bilingual_pairs[idx][0]
        T = self.bilingual_pairs[idx][1]
        
        #E = [self.l1_vocab.token_indices[k] for k in eng_sentence.split(' ')]
        #T = [self.l2_vocab.token_indices[k] for k in tam_sentence.split(' ')]
        
        return np.array(E), np.array(T)

if __name__ == "__main__":

    train_dataset = EnTam("dataset/corpus.bcn.train.en", "dataset/corpus.bcn.train.ta", bucketing_language_sort="l2", cache_id=0)
    vocabs = train_dataset.return_vocabularies()
    word2vecs = train_dataset.return_word2vecs()
    val_dataset = EnTam("dataset/corpus.bcn.dev.en", "dataset/corpus.bcn.dev.ta", bucketing_language_sort="l2", vocabularies=vocabs, word2vecs=word2vecs, cache_id=1)
    #test_dataset = EnTam("dataset/corpus.bcn.test.en", "dataset/corpus.bcn.test.ta", bucketing_language_sort="l2", vocabularies=vocabs, word2vecs=word2vecs, cache_id=2)

    bucketing_batch_sampler = BucketingBatchSampler(val_dataset.bucketer.bucketing_indices, batch_size=16)
    dataloader = DataLoader(val_dataset, batch_sampler=bucketing_batch_sampler)
    #bucketing_batch_sampler = BucketingBatchSampler(val_dataset.bucketer.bucketing_indices, batch_size=16)
    #dataloader = DataLoader(val_dataset, batch_sampler=bucketing_batch_sampler)
    
    for data in dataloader:
    #for data in test_dataset:
        
        l1, l2 = data
        
        #l1, l2 = [l1], [l2]
        for batch in l1:
            print (train_dataset.indices_to_words(batch.tolist(), language="en"))
        for batch in l2:
            print (train_dataset.indices_to_words(batch.tolist(), language="ta"))
        #l1, l2 = l1[0], l2[0]
        print (l1.shape)
        print (l2.shape)
        print ('.'*75)
        print ()
