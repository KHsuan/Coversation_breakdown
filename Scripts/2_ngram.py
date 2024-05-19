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
from functions import *

import warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings("default")

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords') 


from nltk.corpus import stopwords 

# nltk.download()

ngrams_range = (1,1)

#### only do relevance score now
file_name = pd.read_csv("../Data/0_summary_stats_table.csv")
file_name = file_name[file_name["annot_q"].str.contains("relevance", case=False)]

file = file_name["name"].to_list()
path_to_json = '../Data/Target/'
stop_words = set(stopwords.words('english'))
result_all = pd.DataFrame()

for name in file:
    print(name)
    path = path_to_json + name + "_eval.json"
    df = pd.read_json(path)
    
    df_split = relevance_score(df)

    if df_split is None:
        df = df.dropna(subset=['annotations'])
        df_split = relevance_score(df)
    
    df = df_split[["dialogue_id", "context", "response", "overall_score"]]
    text_data = df["response"].tolist()
    df_vec = TMEF_dfm(text_data, ngrams_range = ngrams_range)
    
#     df_vec = df_vec.drop(stop_words, axis=1, errors='ignore')
    scaler = StandardScaler()
    df["overall_score"] = scaler.fit_transform(df[["overall_score"]])

    result_df = prediction_ngram(df_vec, df["overall_score"], name)
    result_all = pd.concat([result_all, result_df], ignore_index=True)

### Change to accuracy(%)
result_all_accuracy = result_all.copy() 
result_all_accuracy.iloc[:, 1:4] = result_all.iloc[:, 1:4].apply(lambda x: 50 * (1 + x))

############################# change path #############################
result_all_accuracy.to_csv("../Data/1_uni_bigram_analysis.csv",index=False)