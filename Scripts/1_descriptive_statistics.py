import os, json
import pandas as pd
import re
from functions import *

def get_basic_stats(df):
    num_turn = len(df)
    annot_q = list(df["annotations"][0].keys())
    num_annot_q = len(annot_q)
    num_annot = len(annot_q)*num_turn
    avg_sentence_context = round(df['sentences_context_num'].mean(),2)
    avg_word_context = round(df['avg_word_context_num'].mean(),2)
    avg_word_per_sentence_context = round(df['avg_word_per_sentence_context_num'].mean(),2)
    avg_sentence_response = round(df['sentences_response_num'].mean(),2)
    avg_word_response = round(df['avg_word_response_num'].mean(),2)
    avg_word_per_sentence_response = round(df['avg_word_per_sentence_response_num'].mean(),2)

    summary_df_single = pd.DataFrame({'num_turn': num_turn,
                                'annot_q':[annot_q],
                                'num_annot_q': num_annot_q,
                                'num_annot': num_annot,
                                'avg_word_context': avg_word_context,
                                'avg_sentence_context': avg_sentence_context,
                                'avg_word_per_sentence_context': avg_word_per_sentence_context,
                                'avg_word_response': avg_word_response,
                                'avg_sentence_response': avg_sentence_response,
                                'avg_word_per_sentence_response': avg_word_per_sentence_response})
    return summary_df_single

def main():
    
    path_to_json = '../Data/Target/'
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    # print(json_files)
    summary_df = pd.DataFrame()
    for i in json_files:
        name = i.replace('_eval.json', '')
        path = path_to_json + i
        df = pd.read_json(path)
        df = df.dropna(subset=['annotations'])
        df = split_word_sentence(df)
        summary_df_single = get_basic_stats(df)
        summary_df_single.insert(0, 'name', name)
        summary_df = pd.concat([summary_df, summary_df_single], axis=0)
    
    print(summary_df.head())
#     summary_df.to_csv("../Data/0_summary_stats_table.csv", index=False)


if __name__ == "__main__":
    main()