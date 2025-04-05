from FeatureGenerator import *
import dill as pickle
import pandas as pd
from nltk.tokenize import sent_tokenize
from helpers import *
import hashlib

"""
Count occurances of selected words in the data files
"""

class CountFeatureGenerator(FeatureGenerator):

    def __init__(self, name = 'CountFeatureGenerator'):
        super().__init__(name)


    def process(self, df):

        grams = ["unigram", "bigram", "trigram"]
        feat_names = ["keyword", "text"]
        print("generate counting features")

        for feat_name in feat_names:
            for gram in grams:             #apply can access multiple columns when analyze a single row (x here)
                df["count_of_%s_%s" % (feat_name, gram)] = list(df.apply(lambda x: len(x[feat_name + "_" + gram]), axis=1))
                df["count_of_unique_%s_%s" % (feat_name, gram)] = \
		            list(df.apply(lambda x: len(set(x[feat_name + "_" + gram])), axis=1))
                df["ratio_of_unique_%s_%s" % (feat_name, gram)] = \
                    list(map(try_divide, df["count_of_unique_%s_%s"%(feat_name,gram)], df["count_of_%s_%s"%(feat_name,gram)]))

        #Overlapping n grams count
        for gram in grams:
            df["count_of_keyword_%s_in_text" % gram] = \
                list(df.apply(lambda x: sum([1. for w in x["keyword_" + gram] if w in set(x["text_" +gram])]), axis = 1))
            df["ratio_of_keyword_%s_in_text" % gram] = \
                list(map(try_divide, df["count_of_keyword_%s_in_text" % gram], df["count_of_keyword_%s" % gram]))

        #finding how many setnences in text
        for feat_name in feat_names:
            df['len_sent_%s' % feat_name] = df[feat_name].apply(lambda x: len(sent_tokenize(x)))

        feat_names = [ n for n in df.columns \
                if "count" in n \
                or "ratio" in n \
                or "len_sent" in n]

        # binary refuting features
        _refuting_words = [
            'fake',
            'fraud',
            'hoax',
            'false',
            'deny', 'denies',
            # 'refute',
            'not',
            'despite',
            'nope',
            'doubt', 'doubts',
            'bogus',
            'debunk',
            'pranks',
            'retract'
        ]

        _hedging_seed_words = [
            'alleged', 'allegedly',
            'apparently',
            'appear', 'appears',
            'claim', 'claims',
            'could',
            'evidently',
            'largely',
            'likely',
            'mainly',
            'may', 'maybe', 'might',
            'mostly',
            'perhaps',
            'presumably',
            'probably',
            'purported', 'purportedly',
            'reported', 'reportedly',
            'rumor', 'rumour', 'rumors', 'rumours', 'rumored', 'rumoured',
            'says',
            'seem',
            'somewhat',
            # 'supposedly',
            'unconfirmed'
        ]

        check_words = _refuting_words

        for rf in check_words:
            fname = '%s_exist' % rf
            feat_names.append(fname)

            df[fname] = df["text"].map(lambda x: 1 if rf in x.lower() else 0)


        print('BasicCountFeatures:')
        print(df)
        train = df.sample(frac=0.8, random_state=2025)
        test = df.loc[~df.index.isin(train.index)]
        n_train = train.shape[0]
        print('BasicCountFeatures, n_train:',n_train)
        n_test = test.shape[0]
        print('BasicCountFeatures, n_test:',n_test)

        xBasicCountsTrain = train[feat_names].values
        outfilename_bcf_train = "train.basic.pkl"
        with open(outfilename_bcf_train, "wb") as outfile:
            pickle.dump(feat_names, outfile, -1)
            pickle.dump(xBasicCountsTrain, outfile, -1)
        print('basic counting features for traning saved in %s' % outfilename_bcf_train)


        print('test:')

        if test.shape[0] > 0:
            print('saving test set')
            xBasicCountsTest = test[feat_names].values
            outfilename_bcf_test = "test.basic.pkl"
            with open (outfilename_bcf_test, 'wb') as outfile:
                pickle.dump(feat_names, outfile, -1)
                pickle.dump(xBasicCountsTest, outfile, -1)
                print('basic counting features for test saved in %s' %outfilename_bcf_test)
        
        return 1
    
    def read(self, header='train'):

        filename_bcf = "%s.basic.pkl" % header
        with open(filename_bcf, "rb") as infile:
            feat_names = pickle.load(infile)
            xBasicCounts = pickle.load(infile)
            print('feature names: ')
            print(feat_names)
            print('xBasicCounts.shape:')
            print(xBasicCounts.shape)      

        return [xBasicCounts]     
    

#“Only run the following code if this script is being run directly (not imported as a module somewhere else).”
if __name__ == '__main__':

    cf = CountFeatureGenerator()
    cf.read()



# Create mock tokenized columns for test
# data = {
#     "text": [
#         "Fake flood warning issued in NYC!",
#         "This is not a hoax.",
#         "Breaking: false alarm"
#     ],
#     "keyword": ["flood", "hoax", "false"],
#     "text_unigram": [["fake", "flood", "warning"], ["not", "a", "hoax"], ["breaking", "false", "alarm"]],
#     "text_bigram": [["fake flood", "flood warning"], ["not a", "a hoax"], ["breaking false", "false alarm"]],
#     "text_trigram": [["fake flood warning"], ["not a hoax"], ["breaking false alarm"]],
#     "keyword_unigram": [["flood"], ["hoax"], ["false"]],
#     "keyword_bigram": [[""]]*3,
#     "keyword_trigram": [[""]]*3,
#     "target": [1, 1, 0]
# }

# df = pd.DataFrame(data)
# cf = CountFeatureGenerator()
#cf.process(df)  # This should create and save train.basic.pkl and test.basic.pkl

# cf.read("train")
# cf.read("test")
