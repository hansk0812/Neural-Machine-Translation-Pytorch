from logger import Logger

import re

class Preprocess(Logger):

    def __init__(self, l1_sentences, l2_sentences):

        self.l1_sentences = l1_sentences
        self.l2_sentences = l2_sentences

    def remove_symbols(self, l1_symbols, l2_symbols):
        
        for idx in range(len(self.l1_sentences)):
            
            re_str_l1 = r"[%s]" % re.escape("".join(l1_symbols))
            re_str_l2 = r"[%s]" % re.escape("".join(l2_symbols))

            self.l1_sentences[idx] = re.sub(re_str_l1, "", self.l1_sentences[idx])
            self.l2_sentences[idx] = re.sub(re_str_l2, "", self.l2_sentences[idx])
