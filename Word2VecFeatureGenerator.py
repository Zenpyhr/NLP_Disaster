from FeatureGenerator import *
import pandas as pd
import numpy as np
import dill as pickle
import gensim
from sklearn.preprocessing import normalize
from functools import reduce
from helpers import *

class Word2VecFeatureGenerator(FeatureGenerator):

    def __init__(self, name = 'word2vecFeatureGenerator'):
        super().__init__(name)   #call the __init__ in parent class
                                 #The subclass would fail to carry over other updates if it set its own name up e.g._name = name

# Test
# w2v = Word2VecFeatureGenerator()
# print(w2v._name)
# print(w2v.name())

    def process(self, df):

        print('generating word2vec features')
        df["text_tokens"] = df["text"].map(lambda x: preprocess_data(x, exclude_stopword=False, stem=False))
        df["keyword_tokens"] = df["keyword"].map(lambda x: preprocess_data(x, exclude_stopword=False, stem=False))

        #Skip the train test dataset split as already provided 
        train = df.sample(frac=0.8, random_state=2025)
        test = df.loc[~df.index.isin(train.index)]


        n_train = train.shape[0]
        print('Word2VecFeatureGenerator, n_train:',n_train)
        n_test = test.shape[0]
        print('Word2VecFeatureGenerator, n_test:',n_test)

        model = gensim.models.KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
        print('Word2Vec model loaded.')

        def get_summed_vector(tokens):
            return reduce(np.add, [model[tok] for tok in tokens if tok in model], np.zeros(300))
        #look up w2v for wach token, sum them up as a single vector

        #processing text
        text_token_array  = df['text_tokens'].values #For debugging?
        print('Text token array:')
        print(text_token_array[:2])
        print(text_token_array.shape)
        print(type(text_token_array ))

        textVec = list(map(get_summed_vector, text_token_array))
        textVec = np.array(textVec)
        textVec = normalize(textVec)
        print("📦 textVec shape:", textVec.shape)

        textVecTrain = textVec[:n_train, :]
        outfilename_textVec_train = "train.text.word2vec.pkl"
        with open(outfilename_textVec_train, "wb") as outfile:
            pickle.dump(textVecTrain, outfile, -1)
        print('text word2vec features of training set saved in ', outfilename_textVec_train)

        if n_test > 0:
            textVecTest = textVec[:n_test, :]
            outfilename_textVec_test = "test.text.word2vec.pkl"
            with open(outfilename_textVec_test, "wb") as outfile:
                pickle.dump(textVecTest, outfile, -1)
            print('text word2vec features of testing set saved in ', outfilename_textVec_test)

        #Processing keyword
        keyword_token_array = df['keyword_tokens'].values
        print('keyword_token_array:')
        print(keyword_token_array[:2])
        print(keyword_token_array.shape)
        print(type(keyword_token_array))

        kwVec = list(map(get_summed_vector, keyword_token_array))
        kwVec = np.array(kwVec)
        kwVec = normalize(kwVec)
        print("📦 kwywrod vector shape:", kwVec.shape)

        kwVecTrain = kwVec[:n_train, :]
        outfilename_kwVec_train = "train.keyword.word2vec.pkl"
        with open(outfilename_kwVec_train, "wb") as outfile:
            pickle.dump(kwVecTrain, outfile, -1)
        print('Keyword word2vec features of training set saved in ', outfilename_kwVec_train)

        if n_test > 0:
            kwVecTest = kwVec[:n_test, :]
            outfilename_kwVec_test = "test.keyword.word2vec.pkl"
            with open(outfilename_kwVec_test, "wb") as outfile:
                pickle.dump(kwVecTest, outfile, -1)
            print('Keyword word2vec features of testing set saved in ', outfilename_kwVec_test)

        print("keyword done")

        #Compute cosine similarity between text/keyword word2vec features

        simVec = np.asarray(list(map(cosine_sim, kwVec, textVec)))[:, np.newaxis]
        #[:, np.newaxis] reshapes it from shape (n,) → (n, 1)


        print('simVec.shape:')
        print(simVec.shape)

        simVecTrain = simVec[:n_train]
        outfilename_simvec_train = "train.sim.word2vec.pkl"
        with open(outfilename_simvec_train, "wb") as outfile:
            pickle.dump(simVecTrain, outfile, -1)
        print('word2vec sim. features of training set saved in ', outfilename_simvec_train)

        if n_test > 0:
            # test set is available
            simVecTest = simVec[n_train:]
            outfilename_simvec_test = "test.sim.word2vec.pkl"
            with open(outfilename_simvec_test, "wb") as outfile:
                pickle.dump(simVecTest, outfile, -1)
            print('word2vec sim. features of test set saved in ', outfilename_simvec_test)

        return 1       
    


    def read(self, dataset_type='train'):
        print(f"Reading Word2Vec features for {dataset_type} set...")

            # Load all .pkl files
        textVec = pickle.load(open(f"{dataset_type}.text.word2vec.pkl", "rb"))
        keywordVec = pickle.load(open(f"{dataset_type}.keyword.word2vec.pkl", "rb"))
        simVec = pickle.load(open(f"{dataset_type}.sim.word2vec.pkl", "rb"))

        print("Shapes:")
        print("textVec:", textVec.shape)
        print("keywordVec:", keywordVec.shape)
        print("simVec:", simVec.shape)

        # Horizontally stack all features together
        full_features = np.hstack([textVec, keywordVec, simVec])
        print(f"Combined features shape: {full_features.shape}")

        return [full_features]


# #test
# # df = pd.read_csv("Data/train.csv")
# # df = df[df["keyword"].apply(lambda x: isinstance(x, str))].copy()
# w2vgen = Word2VecFeatureGenerator()
# # w2vgen.process(df)
# # Load features for training
# X_train = w2vgen.read('train')

# # Load features for test
# X_test = w2vgen.read('test')
# print(X_train)
