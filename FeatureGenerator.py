#Super class of all feature generators

class FeatureGenerator(object):

    def __init__(self, name):
        self._name = name    #Named _name because it's only for internal use
                             #Protected variable
    def name(self):
        return self._name    #Public accessor method



    def process(self, data, header):
        '''
            input:
                data: pandas dataframe
            generate features and save them into a pickle file
        '''
        pass

    def read(self, header):
        '''
            read the feature matrix from a pickle file'
        '''
        
# TEST
# w2v_generator = FeatureGenerator('w2v')
# print(w2v_generator._name)
# print(w2v_generator.name())
