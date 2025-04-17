from FeatureGenerator import *
from TfidfFeatureGenerator import *
import pandas as pd
import numpy as np
from scipy.sparse import vstack
import dill as pickle
from sklearn.decomposition import TruncatedSVD
from helpers import *

class SvdFeatureGenerator(FeatureGenerator):

    def __init__(self, name = 'svdFeatureGenerator'):
        super().__init__(name)

    def process(self,df):
        # train = df.sample(frac = 0.8, random_state = 2025)
        # test = df.loc[~df.index.isin(train.index)]
        # n_train = train.shape[0]
        # print('tfids, n_train:', n_train)
        # n_test = test.shape[0]
        # print('tfidf, n_test:', n_test)




        tfidfGenerator = TfidfFeatureGenerator('tfidf')
        featuresTrain = tfidfGenerator.read('train')
        featuresTest = tfidfGenerator.read('test')

        xKeywordTfidfTrain, xTextTfidfTrain, _ = featuresTrain
        xKeywordTfidfTest, xTextTfidfTest, _ = featuresTest

        n_train = xKeywordTfidfTrain.shape[0]
        n_test = xKeywordTfidfTest.shape[0]
        print("n_train:", n_train)
        print("n_test:", n_test)




        print("xKeywordTfidfTest.shape:", xKeywordTfidfTest.shape)
        print("xTextTfidfTest.shape:", xTextTfidfTest.shape)


        xKeywordTfidf = vstack([xKeywordTfidfTrain, xKeywordTfidfTest])
        xTextTfidf = vstack([xTextTfidfTrain, xTextTfidfTest])
        #stack two sparse matrices vertically
            
        svd = TruncatedSVD(n_components=50, n_iter=15)
        xKTTfidf = vstack([xKeywordTfidf, xTextTfidf])
        svd.fit(xKTTfidf) #Trains the SVD model to find 50 best directions (components) capturing variance in the combined TF-IDF space.
        print('xKeywordTfidf.shape:')
        print(xKeywordTfidf.shape)
        xKeywordSvd = svd.transform(xKeywordTfidf)
        print('xKeywordSvd.shape:')
        print(xKeywordSvd.shape)
        
        xKeywordSvdTrain = xKeywordSvd[:n_train, :]

        outfilename_ksvd_train = "train.keyword.svd.pkl"
        with open(outfilename_ksvd_train, "wb") as outfile:
            pickle.dump(xKeywordSvdTrain, outfile, -1)
        print('keyword svd features of training set saved in ' + str(outfilename_ksvd_train))

        if n_test > 0:
            # test set is available
            xKeywordSvdTest = xKeywordSvd[n_train:, :]
            outfilename_ksvd_test = "test.keyword.svd.pkl"
            with open(outfilename_ksvd_test, "wb") as outfile:
                pickle.dump(xKeywordSvdTest, outfile, -1)
            print('keyword svd features of test set saved in ' + str(outfilename_ksvd_test))

        xTextSvd = svd.transform(xTextTfidf)
        print('xTextSvd.shape:')
        print(xTextSvd.shape)

        xTextSvdTrain = xTextSvd[:n_train, :]
        outfilename_tsvd_train = "train.text.svd.pkl"
        with open(outfilename_tsvd_train, "wb") as outfile:
            pickle.dump(xTextSvdTrain, outfile, -1)
        print('text svd features of training set saved in ' + str(outfilename_tsvd_train))

        print("xTextSvd.shape (all):", xTextSvd.shape)
        print("n_train:", n_train)
        print("xTextSvdTest.shape (slice):", xTextSvd[n_train:, :].shape)



        if n_test > 0:
            # test set is available
            xTextSvdTest = xTextSvd[n_train:, :]
            outfilename_tsvd_test = "test.text.svd.pkl"
            with open(outfilename_tsvd_test, "wb") as outfile:
                pickle.dump(xTextSvdTest, outfile, -1)
            print('text svd features of test set saved in ' + str(outfilename_tsvd_test))

        simSvd = np.asarray(list(map(cosine_sim, xKeywordSvd, xTextSvd)))[:, np.newaxis]
        print('simSvd.shape:')
        print(simSvd.shape)

        simSvdTrain = simSvd[:n_train]
        outfilename_simsvd_train = "train.sim.svd.pkl"
        with open(outfilename_simsvd_train, "wb") as outfile:
            pickle.dump(simSvdTrain, outfile, -1)
        print('svd sim. features of training set saved in ' + str(outfilename_simsvd_train))

        if n_test > 0:
            # test set is available
            simSvdTest = simSvd[n_train:]
            outfilename_simsvd_test = "test.sim.svd.pkl"
            with open(outfilename_simsvd_test, "wb") as outfile:
                pickle.dump(simSvdTest, outfile, -1)
            print('svd sim. features of test set saved in ' + str(outfilename_simsvd_test))

        return 1


    def read(self, header='train'):

        filename_ksvd = "%s.keyword.svd.pkl" % header
        with open(filename_ksvd, "rb") as infile:
            xKeywordSvd = pickle.load(infile)

        filename_tsvd = "%s.text.svd.pkl" % header
        with open(filename_tsvd, "rb") as infile:
            xTextSvd = pickle.load(infile)

        filename_simsvd = "%s.sim.svd.pkl" % header
        with open(filename_simsvd, "rb") as infile:
            simSvd = pickle.load(infile)

        print('xKeywordSvd.shape:')
        print(xKeywordSvd.shape)
        print('xTextSvd.shape:')
        print(xTextSvd.shape)
        print('simSvd.shape:')
        print(simSvd.shape)

        return [xKeywordSvd, xTextSvd, simSvd.reshape(-1, 1)]

# svdgenerator = SvdFeatureGenerator()
# A = svdgenerator.read('train')
# print(A)