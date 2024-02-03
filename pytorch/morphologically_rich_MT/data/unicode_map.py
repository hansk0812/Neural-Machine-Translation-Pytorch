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
    kannada_map = UnicodeMap(language="Kannada", hex_ranges=kannada_hex_ranges, verbose=True)
