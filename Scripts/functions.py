import os, json
import pandas as pd
import re
from scipy.stats import kendalltau 
import nltk
from nltk import bigrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
import pandas as pd
import statistics
import numpy as np
import string
import ssl
from adjustText import adjust_text
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def split_word_sentence(df):

    df['sentences_context'] = df['context'].apply(lambda x: re.split('[.!?]\s+', x))
    df['sentences_context_num'] = df['sentences_context'].apply(lambda x: len(x))

    df['word_context'] = df['context'].apply(lambda x: [word for word in x.split() if not re.match('[^\w\s]', word)])
    df['avg_word_context_num'] = df['word_context'].apply(lambda x: len(x))
    df['avg_word_per_sentence_context_num'] = df['avg_word_context_num'] / df['sentences_context_num']

    df['sentences_response'] = df['response'].apply(lambda x: re.split('[.!?]\s+', x))
    df['sentences_response_num'] = df['sentences_response'].apply(lambda x: len(x))

    df['word_response'] = df['response'].apply(lambda x: [word for word in x.split() if not re.match('[^\w\s]', word)])
    df['avg_word_response_num'] = df['word_response'].apply(lambda x: len(x))
    df['avg_word_per_sentence_response_num'] = df['avg_word_response_num'] / df['sentences_response_num']

    return df

def relevance_score(df):
    
    if any(type(x) == float for x in df['annotations']):
        return None
    else:
        df['overall_score'] = df['annotations'].apply(lambda x: round(statistics.mean(x['relevance']), 2))
    
    return df

def stemming_tokenizer(str_input):
    
    str_input = re.sub('<.*?>', '', str_input)
    # This function will be used to override basic preprocessing steps in TfidfVrctorizer
    snow_stemmer = SnowballStemmer(language='english')
    words = re.sub(r"[^A-Za-z]", " ", str_input).lower().split()
    words = [snow_stemmer.stem(word) for word in words]
    
    return words


def TMEF_dfm(text, ngrams_range = (1,2),
                stop_words = 'english', min_prop = .01,
                max_features=None):

  # TfidfVectorizer and CountVectorizer removes punctuation automatically
  # we also pass an earlier stemming_tokenizer function to the text
  # stopword options are either 'english' or False currently
    
    if stop_words == 'english':
        vec = CountVectorizer(
            tokenizer = stemming_tokenizer,
            stop_words = stop_words,
            ngram_range=ngrams_range,
            min_df=min_prop,
            max_features=max_features,
            token_pattern='(?u)\\b\\w+\\b'
            )
    else:
        vec = CountVectorizer(
            tokenizer = stemming_tokenizer,
            ngram_range=ngrams_range,
            min_df=min_prop,
            max_features=max_features,
            token_pattern='(?u)\\b\\w+\\b'
        )
    
    X = vec.fit_transform(text)
    
    df = round(pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out()),2)
    
    return df

def prediction_ngram(X, Y, name = "model"):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    mse_values = []
    for c in np.linspace(0.001, 1, 100):
        lasso = Lasso(alpha=c)
        cv_scores = cross_val_score(lasso, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
        mse_values.append(-1 * np.mean(cv_scores))

    best_alpha = np.linspace(0.001, 1, 100)[np.argmin(mse_values)]

    best_lasso = Lasso(alpha=best_alpha).fit(X_train, y_train)
    predict_lasso = best_lasso.predict(X_test)
    lasso_tau, _ = kendalltau(y_test, predict_lasso)

    regressor = RandomForestRegressor(random_state=42)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    rf_tau, _ = kendalltau(y_test, y_pred)

    regressor = HistGradientBoostingRegressor(random_state=42)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    xgb_tau, _ = kendalltau(y_test, y_pred)


    result_df = pd.DataFrame({
        'Dataset': [name],
        'Lasso_kendalltau': [lasso_tau],
        'Random_Forest_kendalltau': [rf_tau],
        'XGBoost_kendalltau': [xgb_tau]
    })


    return result_df