from FeatureGenerator import *
import pandas as pd
import numpy as np 
import dill as pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from helpers import *


class SentimentFeatureGenerator(FeatureGenerator):

    def __init__(self, name = 'sentimentFeatureGenerator'):
        super().__init__(name)

    def process(self, df):

        print('generating sentiment features')
        print('for keyword')

        train = df.sample(frac=0.8, random_state = 2025)
        test = df.loc[~df.index.isin(train.index)]
        n_train = train.shape[0]
        print(n_train)
        n_test = test.shape[0]       

        #calculate the polarity score of each sentence then take the average
        sid = SentimentIntensityAnalyzer()
        def compute_sentiment(sentences):   #aggregate polarity and take mean
            result = []
            for sentence in sentences:
                vs = sid.polarity_scores(sentence)
                result.append(vs)
            return pd.DataFrame(result).mean()
        

        print('For keyword')

        analyzer = SentimentIntensityAnalyzer()   #directly use VADER’s polarity_scores()
        df['keyword_sents'] = df['keyword'].apply(lambda x: analyzer.polarity_scores(str(x)))
        df = pd.concat([df, df['keyword_sents'].apply(pd.Series)], axis = 1)
        df.rename(columns={'compound':'k_compound', 'neg':'k_neg', 'neu':'k_neu', 'pos':'k_pos'}, inplace=True)
        
        print(df)
        
        keywordSenti = df[['k_compound','k_neg','k_neu','k_pos']].values
        print('keywordSenti.shape:')
        print(keywordSenti.shape)

        keywordSentiTrain = keywordSenti[:n_train, :]
        outfilename_ksenti_train = "train.keyword.senti.pkl"
        with open(outfilename_ksenti_train, "wb") as outfile:
            pickle.dump(keywordSentiTrain, outfile, -1)
        print('keyword sentiment features of training set saved in ', outfilename_ksenti_train)

        if n_test > 0:
            keywordSentiTest = keywordSenti[n_train:, :]
            outfilename_ksenti_test = "test.keyword.senti.pkl"
            with open(outfilename_ksenti_test, "wb") as outfile:
                pickle.dump(keywordSentiTest, outfile, -1)
            print('keyword sentiment features of test set saved in ', outfilename_ksenti_test)

        print('keyword senti done')



        print('For text')
        df['text_sents'] = df['text'].map(lambda x: sent_tokenize(str(x)) if pd.notnull(x) else [])
        df = pd.concat([df, df['text_sents'].apply(lambda x: compute_sentiment(x))], axis=1)
        df.rename(columns={'compound':'t_compound', 'neg':'t_neg', 'neu':'t_neu', 'pos':'t_pos'}, inplace=True)

        textSenti = df[['t_compound','t_neg','t_neu','t_pos']].values
        print('textSenti.shape:')
        print(textSenti.shape)

        textSentiTrain = textSenti[:n_train, :]
        outfilename_tsenti_train = "train.text.senti.pkl"
        with open(outfilename_tsenti_train, "wb") as outfile:
            pickle.dump(textSentiTrain, outfile, -1)
        print('text sentiment features of training set saved in ', outfilename_tsenti_train)

        if n_test > 0:
            textSentiTest = textSenti[n_train:, :]
            outfilename_tsenti_test = "test.text.senti.pkl"
            with open(outfilename_tsenti_test, "wb") as outfile:
                pickle.dump(textSentiTest, outfile, -1)
            print('text sentiment features of test set saved in ', outfilename_tsenti_test)

        print('text senti done')

        return 1

    def read(self, header='train'):

        filename_ksenti = "%s.keyword.senti.pkl" % header
        with open(filename_ksenti, "rb") as infile:
            keywordSenti = pickle.load(infile)

        filename_tsenti = "%s.text.senti.pkl" % header
        with open(filename_tsenti, "rb") as infile:
            textSenti = pickle.load(infile)

        print('keywordSenti.shape:')
        print(keywordSenti.shape)
        print('textSenti.shape:')
        print(textSenti.shape)

        return [keywordSenti, textSenti]



# # Sample dummy data
# data = {
#     "keyword": [
#         "earthquake",
#         "fire",
#         "flood"
#     ],
#     "text": [
#         "A strong earthquake shook the city this morning. People ran out in panic.",
#         "Firefighters are battling a massive blaze in the downtown area.",
#         "Heavy rains caused flash flooding in several areas, damaging property and vehicles."
#     ]
# }

# # Create DataFrame
# df = pd.DataFrame(data)

# # Initialize and run the sentiment feature generator
# sentiGen = SentimentFeatureGenerator()
# sentiGen.process(df)

# # Read the generated features
# keyword_senti, text_senti = sentiGen.read('train')

# # Print shapes and contents
# print("\n📦 Keyword Sentiment Features (train):")
# print(keyword_senti.shape)
# print(keyword_senti)

# print("\n📦 Text Sentiment Features (train):")
# print(text_senti.shape)
# print(text_senti)

