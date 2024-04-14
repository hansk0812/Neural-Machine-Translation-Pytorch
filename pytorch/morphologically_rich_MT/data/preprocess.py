from .logger import Logger
from unidecode import unidecode

import re
import string
import stanza
try:
    stanza.download('en')
    #stanza.download('ta')
    en_nlp = stanza.Pipeline('en', processors='tokenize', download_method=None)
    #ta_nlp = stanza.Pipeline('ta', processors='tokenize')
except ConnectionError:
    en_nlp = stanza.Pipeline('en', processors='tokenize', download_method=None)
    #ta_nlp = stanza.Pipeline('ta', processors='tokenize', download_method=None)


class Preprocess(Logger):

    def __init__(self, l1_sentences, l2_sentences, verbose=False):

        super().__init__(verbose)

        assert len(l1_sentences) == len(l2_sentences)

        self.l1_sentences = l1_sentences
        self.l2_sentences = l2_sentences

    def remove_symbols(self, l1_symbols, l2_symbols, replaceable_symbols):

        translator = str.maketrans('', '', string.punctuation) #TODO re bug
        for idx in range(len(self.l1_sentences)):
            
            re_str_l1 = "[%s]" % re.escape("".join(l1_symbols))
            re_str_l2 = "[%s]" % re.escape("".join(l2_symbols))

            self.l1_sentences[idx] = re.sub(re_str_l1, "", self.l1_sentences[idx])
            self.l2_sentences[idx] = re.sub(re_str_l2, "", self.l2_sentences[idx])
            
            self.l1_sentences[idx] = self.l1_sentences[idx].translate(translator)
            
            for rep in replaceable_symbols:
                self.l1_sentences[idx] = ' '.join(self.l1_sentences[idx].replace(rep, replaceable_symbols[rep]).split())
                self.l2_sentences[idx] = ' '.join(self.l2_sentences[idx].replace(rep, replaceable_symbols[rep]).split())

    def lower_case_english(self, language="l1"):
        
        assert language in ["l1", "l2"]
        
        if language == "l1":
            self.l1_sentences = [x.lower() for x in self.l1_sentences]
        else:
            self.l2_sentences = [x.lower() for x in self.l2_sentences]

    def unidecode_english(self, sentences, reserved_num="NUM"):

        for idx in range(len(sentences)):
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
            
            # c in francois
            sentences[idx] = sentences[idx].replace("aSS", "c")
        
            # manual change
            num_and_a_half = lambda x: "%s%s" % (reserved_num, x) # NUM a half --> NUM and a half
            sentences[idx] = sentences[idx].replace(num_and_a_half(" a 1/2"), num_and_a_half(" and a half"))
        
        return sentences

    def reserved_token_num(self, reserved_num_token):
    
        for idx in range(len(self.l1_sentences)):
            self.l1_sentences[idx] = re.sub(r"[0-9]+[\.0-9]*", reserved_num_token, self.l1_sentences[idx])
            self.l2_sentences[idx] = re.sub(r"[0-9]+[\.0-9]*", reserved_num_token, self.l2_sentences[idx])

    def tokenize_english(self):

        for idx, sentence in enumerate(self.l1_sentences):
            doc = en_nlp(sentence)
            if len(doc.sentences) > 1:
                tokens = [x.text for x in doc.sentences[0].tokens]
                for sent in doc.sentences[1:]:
                    tokens.extend([x.text for x in sent.tokens])
            else:
                try:
                    tokens = [x.text for x in doc.sentences[0].tokens]
                except IndexError:
                    tokens = []

            self.l1_sentences[idx] = ' '.join(tokens)
 
