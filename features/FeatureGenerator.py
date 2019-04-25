import pickle

class FeatureGenerator:
    
    def __init__(self, name):
        self._name = name
    

    def process(self, data, header):
        pass

    
    def read(self, header):
        pass

    def _dump(self, df, filename):
        print(filename, "--- Shape:", df.shape)
        with open(filename, 'wb') as outfile:
            pickle.dump(df, outfile, -1)