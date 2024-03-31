from logger import Logger

class WordEmbedding(Logger):

    def __init__(self, bilingual_pairs, save_dir, monolingual_l1, monolingual_l2, word_vector_size,
                 identifier="entam", symbols=False, morphemes=False, verbose=False, split="train"):
        
        super().__init__(verbose)

        assert len(languages) == 2
        self.word_vector_size = word_vector_size
        self.morphemes = morphemes
        self.identifier = identifier
        self.split = split
        
        assert split in ["train", "val", "test"]

        dir_name = "tokens" if not morphemes else "morphemes"
        dir_name += "_symbols" if symbols else "_nosymbols"

        if (os.path.exists(os.path.join(save_dir, dir_name, "word2vec_%s.l1.model" % identifier)) and \
           os.path.exists(os.path.join(save_dir, dir_name, "word2vec_%s.l2.model" % identifier))) or split != "train":
            self.l1_wv = Word2Vec.load(os.path.join(save_dir, "tokens" if not morphemes else "morphemes", "word2vec_%s.l1.model" % identifier))
            self.l2_wv = Word2Vec.load(os.path.join(save_dir, "tokens" if not morphemes else "morphemes", "word2vec_%s.l2.model" % identifier))
        else:
            self.l1_wv, self.l2_wv = None, None
            save_folder = os.path.join(save_dir, dir_name)

            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
            
            word2vec_l1_sentences = [x[0] for x in bilingual_pairs] + monolingual_l1
            word2vec_l2_sentences = [x[1] for x in bilingual_pairs] + monolingual_l2
            self.train_word2vec_model_on_monolingual_and_mt_corpus(word2vec_l1_sentences, language_id=0, save_folder=save_folder)
            self.train_word2vec_model_on_monolingual_and_mt_corpus(word2vec_l2_sentences, language_id=1, save_folder=save_folder)
        
        self.l1_embedding = np.array([self.get_word2vec_embedding_for_token(word, language_id=0) for word in self.eng_vocabulary])
        self.l2_embedding = np.array([self.get_word2vec_embedding_for_token(word, language_id=1) for word in self.tam_vocabulary])
 
    def train_word2vec_model_on_monolingual_and_mt_corpus(self, sentences, language_id, save_folder):
    
        assert language_id in [0, 1]
        
        if language_id == 0:
            self.l1_wv = Word2Vec(sentences=sentences, vector_size=self.word_vector_size, window=5, min_count=1, workers=4)
            self.l1_wv.build_vocab(sentences)
            self.l1_wv.train(sentences, total_examples=len(sentences), epochs=20)
            self.l1_wv.save(os.path.join(save_folder, "word2vec_%s.l1.model" % (self.identifier)))

        else:
            self.l2_wv = Word2Vec(sentences=sentences, vector_size=self.word_vector_size, window=5, min_count=1, workers=4)
            self.l2_wv.build_vocab(sentences)
            self.l2_wv.train(sentences, total_examples=len(sentences), epochs=20)
            self.l2_wv.save(os.path.join(save_folder, "word2vec_%s.l2.model" % (self.identifier)))

        self.print("Word2vec model trained successfully for language: %d" % language_id)
    
    def get_word2vec_embedding_for_token(self, token, lang, unknown_token="UNK"):
        
        assert lang in [0,1]

        try:
            if lang == 0:
                return self.l1_wv.wv[token]
            else:
                return self.l2_wv.wv[token]
        
        except (KeyError, AttributeError):
            
            #traceback.print_exc()
            self.print("Token not in %s language %d word2vec vocabulary: %s" % (self.split, lang, token))
            # word vector not in vocabulary - possible for tokens in val and test sets
            if lang == "en":
                return self.l1_wv.wv[unknown_token]
            else:
                return self.l2_wv.wv[unknown_token]
    
    def create_spl_token_sequences_for_word2vec(self, unknown="UNK", num="NUM", eng="ENG", pad="PAD", bos="START", eos="END"):
        #TODO
        pass
        
