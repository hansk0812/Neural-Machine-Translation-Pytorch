from preprocess import Preprocess
from bucket import Bucketing
from unicode_map import UnicodeMap

from cache import Cache
from vocabulary import Vocabulary
from word_embedding import WordEmbedding

from utils import get_sentences_from_file

import string

from torch.utils.data import Dataset

class EnTam(Dataset):

    VERBOSE = True
    
    tamil_hex_ranges = [("b82", 2), ("b85", 6), ("b8e", 3), 
                        ("b92", 4), ("b99", 2), ("b9c", 1),
                        ("b9e", 2), ("ba3", 2), ("ba8", 3),
                        ("bae", 12), ("bbe", 5), ("bc6", 3),
                        ("bca", 4), ("bd0", 1), ("bd7", 1),
                        ("be6", 21)]

    reserved_tokens = ["UNK", "PAD", "ENG", "NUM", "START", "END"]

    def __init__(self, l1_fpath, l2_fpath, start_stop=False):

        super().__init__()

        l1_sentences, l2_sentences = get_sentences_from_file(l1_fpath, l2_fpath)

        #...........................................................................................................................................        
        # Preprocessing + Caching
        #...........................................................................................................................................        

        preprocessing_cache = Cache("cache", cache_id=0)
        if not preprocessing_cache.is_file("tokenized.en") or preprocessing_cache.is_file("tokenized.ta"):

            unnecessary_symbols = ["¦", "¡", "¬", '\u200c']
            symbol_replacements = {"‘": "'", '“': '"', '”': "\"", "’": "'"}

            # Preprocess
            self.preprocess = Preprocess(l1_sentences, l2_sentences, verbose=self.VERBOSE)
            self.preprocess.remove_symbols(list(string.punctuation) + unnecessary_symbols, list(string.punctuation) + unnecessary_symbols, symbol_replacements)
            self.preprocess.lower_case_english("l1")
            self.preprocess.reserved_token_num(self.reserved_tokens[3])
            self.preprocess.l1_sentences = self.preprocess.unidecode_english(self.preprocess.l1_sentences)
            #self.preprocess.l2_sentences = self.preprocess.unidecode_english(self.preprocess.l2_sentences) #transliteration
        
            self.preprocess.tokenize_english(language=1)

            # Unicode code block restrictions
            tamil_map = UnicodeMap(language="Tamil", hex_ranges=self.tamil_hex_ranges, verbose=self.VERBOSE)
            for l2_idx in range(len(self.preprocess.l2_sentences)):
                sentence_tokens = []
                for token in self.preprocess.l2_sentences[l2_idx].split(' '):
                    sentence_tokens.extend(tamil_map.tokenize(token, self.reserved_tokens, eng_token="ENG"))
                self.preprocess.l2_sentences[l2_idx] = ' '.join(sentence_tokens)
        
            preprocessing_cache.variable_to_file(self.preprocess.l1_sentences, "tokenized.en")
            preprocessing_cache.variable_to_file(self.preprocess.l2_sentences, "tokenized.ta")
        
        else: # use cached preprocessed sentences
            
            self.preprocess.l1_sentences = preprocessing_cache.file_to_variable("tokenized.en")
            self.preprocess.l2_sentences = preprocessing_cache.file_to_variable("tokenized.ta")
        
        #...........................................................................................................................................        
        # Preprocessing + Caching
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

if __name__ == "__main__":

    dataset = EnTam("../dataset/corpus.bcn.train.en", "../dataset/corpus.bcn.train.ta")

    idx = 500
    for l1, l2 in zip(dataset.preprocess.l1_sentences, dataset.preprocess.l2_sentences):
        print (l1)
        print ()
        print (l2)
        print ('.'*75)
        print ()
        if idx == 0:
            break
        idx -= 1
