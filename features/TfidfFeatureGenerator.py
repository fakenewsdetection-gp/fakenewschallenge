from FeatureGenerator import FeatureGenerator

class TfidfFeatureGenerator(FeatureGenerator):
    
    def __init__(self, name='tfidf'):
        super(TfidfFeatureGenerator, self).__init__(name)


    def process(self, df):
        pass


    def read(self, header='train'):
        pass