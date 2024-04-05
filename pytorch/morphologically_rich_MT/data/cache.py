import os
from .logger import Logger

class Cache(Logger):

    def __init__(self, cache_dir, verbose=False, cache_id=0):
        
        super().__init__(verbose)
        self.cache_id = cache_id
        self.cache_dir = os.path.join(cache_dir, str(cache_id))

        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

    def cache_file(self, fname, list_of_str, append=False):

        mode = "w" if not append else "a"

        with open(os.path.join(self.cache_dir, fname), mode) as f:
            for idx, unit in enumerate(list_of_str):
                if idx + 1 ==len(list_of_str):
                    f.write(unit)
                else:
                    f.write(unit + '\n')

    def file_to_variable(self, fname, dtype="list"):
        if dtype == "wv":
            raise NotImplementedError
        elif dtype == "list":
            with open(os.path.join(self.cache_dir, fname), 'r') as f:
                var = [x.strip() for x in f.readlines()]
            return var 

    def variable_to_file(self, variable, fname, dtype="list"):
        if not dtype == "list":
            raise NotImplementedError
        self.cache_file(fname, variable, append=False)

    def is_file(self, fname):
        return os.path.exists(os.path.join(self.cache_dir, fname))
    
    def get_path(self, fname):
        return os.path.join(self.cache_dir, fname)

if __name__ == "__main__":

    x = ["test4", "test5", "test6"]   

    cache = Cache("dir")
    cache.variable_to_file(x, "test.txt")
    var = cache.file_to_variable("test.txt")

    print (var)
