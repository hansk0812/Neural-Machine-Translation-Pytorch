from data.logger import Logger
from data.cache import Cache
from data.utils import get_sentences_from_file

reserved_tokens = ["UNK", "PAD", "ENG", "NUM", "START", "END"]

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
