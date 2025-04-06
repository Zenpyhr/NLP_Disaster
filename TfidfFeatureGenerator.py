from FeatureGenerator import *
import pandas as pd
import numpy as np
import dill as pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from helpers import *

class TfidfFeatureGenerator(FeatureGenerator):

    def __init__(self, name = 'tfidfFeatureGenerator'):
        super().__init__(name)

    def process(self, df):

        #1.create strings based on ' '.join(keyword_unigram + text_unigram) (stemmed)
        def cat_text(x):
            return ' '.join(x['keyword_unigram'] + x['text_unigram'])

        df["all_text"] = list(df.apply(cat_text, axis = 1))

        train = df.sample(frac=0.8, random_state=2025)
        test = df.loc[~df.index.isin(train.index)]       
        n_train = train.shape[0]
        print('tfidf, n_train:',n_train)
        n_test = test.shape[0]
        print('tfidf, n_test:',n_test)

        #build the vocab not actually transform anything into a tf-idf matrix
        vec = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2)
        vec.fit(df["all_text"]) # Tf-idf calculated on the combined training + test set
        vocabulary = vec.vocabulary_

        #for keyword
        vecK = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2, vocabulary=vocabulary) #freeze the vocab always use the joint vocabulary
        xKeywordTfidf = vecK.fit_transform(df['keyword_unigram'].map(lambda x: ' '.join(x)))

        print('xKeywordTfidf.shape:')

        print(xKeywordTfidf.shape)

        xKeywordTfidfTrain = xKeywordTfidf[:n_train, :]
        outfilename_ktfidf_train = "train.keyword.tfidf.pkl"
        with open(outfilename_ktfidf_train, "wb") as outfile:
            pickle.dump(xKeywordTfidfTrain, outfile, -1)
        print('keyword tfidf features of training set saved in ', outfilename_ktfidf_train)

        if n_test > 0:
            # test set is available
            xKeywordTfidfTest = xKeywordTfidf[n_train:, :]
            outfilename_ktfidf_test = "test.keyword.tfidf.pkl"
            with open(outfilename_ktfidf_test, "wb") as outfile:
                pickle.dump(xKeywordTfidfTest, outfile, -1)
            print('keyword tfidf features of test set saved in ', outfilename_ktfidf_test)


        #for text
        vecT = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2, vocabulary=vocabulary)
        xTextTfidf = vecT.fit_transform(df['text_unigram'].map(lambda x: ' '.join(x)))
        print('xTextTfidf.shape:')
        print(xTextTfidf.shape)

        # Save train and test into separate files
        xTextTfidfTrain = xTextTfidf[:n_train, :]
        outfilename_ttfidf_train = "train.text.tfidf.pkl"
        with open(outfilename_ttfidf_train, "wb") as outfile:
            pickle.dump(xTextTfidfTrain, outfile, -1)
        print('text tfidf features of training set saved in ', outfilename_ttfidf_train)

        if n_test > 0:
            # test set is available
            xTextTfidfTest = xTextTfidf[n_train:, :]
            outfilename_ttfidf_test = "test.text.tfidf.pkl"
            with open(outfilename_ttfidf_test, "wb") as outfile:
                pickle.dump(xTextTfidfTest, outfile, -1)
            print('text tfidf features of test set saved in ', outfilename_ttfidf_test)

        #4.compute cosine similarity between keyword tfidf features and text tfidf features

        simTfidf = np.array(list(map(cosine_sim, xKeywordTfidf, xTextTfidf)))[:, np.newaxis]
        print('simTfidf.shape')
        print(simTfidf.shape)
        simTfidfTrain = simTfidf[:n_train]
        outfilename_simtfidf_train = "train.sim.tfidf.pkl"

        with open(outfilename_simtfidf_train, "wb") as outfile:
            pickle.dump(simTfidfTrain, outfile, -1)
        print('tfidf sim. features of training set saved in', outfilename_simtfidf_train)

        # Save test part
        if n_test > 0:
            simTfidfTest = simTfidf[n_train:]
            outfilename_simtfidf_test = "test.sim.tfidf.pkl"
            with open(outfilename_simtfidf_test, "wb") as outfile:
                pickle.dump(simTfidfTest, outfile, -1)
            print('tfidf sim. features of test set saved in', outfilename_simtfidf_test)

        return 1
    
    def read(self, header='train'):
        filename_ktfidf = f"{header}.keyword.tfidf.pkl"
        with open(filename_ktfidf, "rb") as infile:
            xKeywordTfidf = pickle.load(infile)

        filename_ttfidf = f"{header}.text.tfidf.pkl"
        with open(filename_ttfidf, "rb") as infile:
            xTextTfidf = pickle.load(infile)

        filename_simtfidf = f"{header}.sim.tfidf.pkl"
        with open(filename_simtfidf, "rb") as infile:
            simTfidf = pickle.load(infile)

        print("xKeywordTfidf.shape:")
        print(xKeywordTfidf.shape)
        print("xTextTfidf.shape:")
        print(xTextTfidf.shape)
        print("simTfidf.shape:")
        print(simTfidf.shape)

        # Return just the similarity, or all features if needed
        return [xKeywordTfidf, xTextTfidf, simTfidf.reshape(-1, 1)]
        # return [xKeywordTfidf, xTextTfidf, simTfidf.reshape(-1, 1)]  # optional full return



# # test
# df = pd.read_csv("Data/train.csv")

# # Filter out rows with missing keywords (optional for testing)
# df = df[df["keyword"].apply(lambda x: isinstance(x, str))].copy()

# # Add n-grams to match what your process function expects
# for col in ["text", "keyword"]:
#     df[f"{col}_unigram"] = df[col].map(lambda x: preprocess_data(x, exclude_stopword=False, stem=True))
#     df[f"{col}_bigram"] = df[f"{col}_unigram"].map(lambda tokens: list(zip(tokens, tokens[1:])))
#     df[f"{col}_trigram"] = df[f"{col}_unigram"].map(lambda tokens: list(zip(tokens, tokens[1:], tokens[2:])))

# # Create and test the feature generator
# tfidf_gen = TfidfFeatureGenerator()
# tfidf_gen.process(df)

# Read and inspect the result
# features = tfidf_gen.read('train')
# print("Sample TF-IDF cosine sim features from training:")
# print(features[0][:5])  # First 5 cosine similarity values