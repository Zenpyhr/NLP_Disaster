import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

english_stemmer = nltk.stem.SnowballStemmer('english')
token_pattern = r"(?u)\b\w\w+\b"
stopwords = set(nltk.corpus.stopwords.words('english'))


def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed

# test 
#entence = "You're having a great day easily finished my homework!"  
# token_example = word_tokenize(sentence)
# print(stem_tokens(token_example, english_stemmer))

def preprocess_data(line,
                    token_pattern = token_pattern,  
                    # the regex pattern used to split the text into words
                    exclude_stopword = True,
                    stem = True):
    token_pattern = re.compile(token_pattern, flags = re.UNICODE)
    #re.compile change the string pattern e.g.r"(?u)\b\w\w+\b" into serach tool 
    tokens = [x.lower() for x in token_pattern.findall(line)] 
     #Finds all parts of the text that match the pattern

    tokens_stemmed = tokens
    if stem:
        tokens_stemmed = stem_tokens(tokens, english_stemmer)
    if exclude_stopword:
        tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords] 
    
    return tokens_stemmed


# print(preprocess_data(sentence))


def try_divide(x, y, val = 0.0):
    """
        Try to divide two numbers
    """
    if y != 0.0:
        val = float(x) / y
    return val

def cosine_sim(x,y):
    try:
        if type(x) is np.ndarray: x = x.reshape(1, -1)
        if type(y) is np.ndarray: y = y.reshape(1, -1)
        d = cosine_similarity(x, y)
        d = d[0][0]
    except:
        print(x)
        print(y)
        d = 0.
    return d
    

