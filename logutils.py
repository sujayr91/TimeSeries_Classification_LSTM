class logger(object):
    def __init__(self):
        self.log= []
        self.itr= []

    def reset(self):
        self.log= []
        self.itr= []

    def update(self,logval, itrval):
        self.log+=[logval]
        self.itr+=[itrval]
