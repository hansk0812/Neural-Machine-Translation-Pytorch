class Logger:

    def __init__(self, verbose):
        self.verbose = verbose
    
    def print(self, x):
        if self.verbose:
            print (x)
