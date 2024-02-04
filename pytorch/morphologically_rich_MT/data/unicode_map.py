import string
from collections import OrderedDict

from logger import Logger

class UnicodeMap(Logger):

    def __init__(self, language, hex_ranges, verbose):
        
        # The logger simply uses the verbose flag around print statements. 
        # This class should be plug-and-play if you replace its self.print statements with print()
        super().__init__(verbose)

        self.language = language

        self.map = []
        for hex_start, hex_range in hex_ranges:
            self.map.extend(self.return_unicode_hex_within_range(hex_start, hex_range))

    def check_unicode_block(self, character):
        # Returns True if `character` is inside `unicode_block`

        unicode_hex = "".join("{:02x}".format(ord(x)) for x in character)
        return unicode_hex in self.map
    
    def return_unicode_hex_within_range(self, start, num_chars):
        assert isinstance(start, str) and isinstance(num_chars, int)
        start_integer = int(start, 16)
        return ["".join("{:02x}".format(start_integer+x)).lower() for x in range(num_chars)]
    
    def get_membership(self, word):
        
        # Returns `True` if all characters in `word` belong to this unicode map block
        for character in word:
            if not self.check_unicode_block(character):
                return False
        return True
    
    def to_lower(self, sentence, mappings):
        
        for character in sentence:
            unicode_hex = "".join("{:02x}".format(ord(x)) for x in character)
            assert unicode_hex in mappings.keys() or unicode_hex in mappings.values()

        return

    def get_en_unicode_sequence(self, token):
    
        # Takes a token with english characters and unicode characters and returns a sequence dictionary for tokenization
        sequence = OrderedDict()
        num_eng, num_unicode = 0, 0
        get_count = lambda lang: str(num_eng) if lang=='en' else str(num_unicode)

        if self.check_unicode_block(token[0]):
            lang = 'uc'
            num_unicode += 1
        else:
            lang = 'en'
            num_eng += 1

        sequence[lang+"0"] = 0

        for idx, character in enumerate(list(token)[1:]):

            if self.check_unicode_block(character):
                if lang == 'en':
                    lang = 'uc'
                    sequence[lang+get_count(lang)] = idx + 1
                    num_unicode += 1
            else:
                if lang == 'uc':
                    lang = 'en'
                    sequence[lang+get_count(lang)] = idx + 1
                    num_eng += 1

        sequence[lang+get_count(lang)] = len(token)
        return sequence

    def tokenize_en_unicode_combinations(self, token_languages, token, english_reserved_token="ENG"):
        
        # token_languages: Return dict of self.get_en_unicode_sequence
        # token: Word to be split by language of characters
        # english_reserved_token: Reserved word to replace >=3 consecutive english characters with

        tokens_split, unicode_part = [], ""
        keys = list(token_languages.keys())
        for idx, key in enumerate(reversed(keys[:-1])):
            lang = "en" if "en" in key else "uc"
            start_of_lang_block = token_languages[key]
            end_of_lang_block = token_languages[keys[len(keys)-1 - idx]]
            
            if lang == "en":
                if end_of_lang_block - start_of_lang_block >= 3:
                    if unicode_part == "":
                        tokens_split.append(english_reserved_token)
                    else:
                        tokens_split.extend([unicode_part, english_reserved_token])
                        unicode_part = ""
            else:
                unicode_part = token[start_of_lang_block:end_of_lang_block] + unicode_part
        
        if unicode_part != "":
            tokens_split.append(unicode_part)
        else:
            # no unicode block characters means <=2 character english token
            tokens_split.append(english_reserved_token)

        tokens_split = list(reversed(tokens_split))

        return tokens_split

    def return_en_unicode_tokenized_sentence(self, sentence, english_reserved_token):

        # sentence: Unicode sentence to apply tokenization with
        # english_reserved_token: Token to replace >=3 consecutive english characters

        tokens = sentence.split(' ')
        
        for token_index, token in enumerate(tokens):

            if not token in string.punctuation:
                
                token_languages = self.get_en_unicode_sequence(token)
                token_replacement = self.tokenize_en_unicode_combinations(token_languages, token, \
                                            english_reserved_token=english_reserved_token)
                
                if token_replacement[0] == token:
                    continue
                else:
                    new_sentence_tokens = tokens[:token_index] + token_replacement + tokens[token_index+1:]
                    sentence = " ".join(new_sentence_tokens)
        
        return sentence

if __name__ == "__main__":

    tamil_hex_ranges = [("b82", 2), ("b85", 6), ("b8e", 3), 
                        ("b92", 4), ("b99", 2), ("b9c", 1),
                        ("b9e", 2), ("ba3", 2), ("ba8", 3),
                        ("bae", 12), ("bbe", 5), ("bc6", 3),
                        ("bca", 4), ("bd0", 1), ("bd7", 1),
                        ("be6", 21)]

    kannada_hex_ranges = [("c80", 13), ("c8e", 3), ("c8e", 23),
                          ("caa", 10), ("cbc", 9), ("cc6", 3),
                          ("cca", 4), ("cd5", 2), ("cdd", 2),
                          ("ce0", 4), ("ce6", 10), ("cf1", 3)]

    tamil_map = UnicodeMap(language="Tamil", hex_ranges=tamil_hex_ranges, verbose=True)
    sentence = "ஆதasdgலால் அந்த வாdலிபsரின் பாவம் கர்dadgத்தருfdsasfடைய சந்நிfaaதியில் மிகsவுdம் பெaddரிsfaதாsயிdருeந்fதது."
    tokenized = tamil_map.return_en_unicode_tokenized_sentence(sentence, "ENG")
    print (sentence, tokenized)

    #kannada_map = UnicodeMap(language="Kannada", hex_ranges=kannada_hex_ranges, verbose=True)
