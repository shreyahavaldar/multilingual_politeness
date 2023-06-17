import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import argparse
import time, sys
import re


start_categories = ["please_start", "direct_start", "first_person_start", "second_person_start"]

def construct_regex(category, lexica_df, word_bound=True, start=False):
    lexica = lexica_df[lexica_df["CATEGORY"] == category]["word"].tolist()
    lexica = [str(x) for x in lexica]
    if(not start and word_bound):
        regex = re.compile(r'\b(%s)\b' % '|'.join(lexica), re.IGNORECASE)
    if(start and word_bound):
        #only match if word is at the start of the sentence
        regex = re.compile(r'(^|[.?!][ ])(%s)\b' % '|'.join(lexica), re.IGNORECASE)
    if(not word_bound):
        regex = re.compile(r'(%s)' % '|'.join(lexica), re.IGNORECASE)
    return regex

def construct_politeness_strategy_regexes(lexica_df, language):
    categories = lexica_df["CATEGORY"].unique().tolist()
    regexes = {}
    word_bound = True
    if(language == "Chinese" or language == "Japanese"):
        word_bound = False
    for category in categories:
        start = False
        if(category in start_categories):
            start = True
        regexes[category] = construct_regex(category, lexica_df, word_bound, start)
    return regexes

def load_lexica():
    en_lex = pd.read_csv('../MLG/Lexica/english_politelex.csv')
    es_lex = pd.read_csv('../MLG/Lexica/spanish_politelex_purified.csv')
    ja_lex = pd.read_csv('../MLG/Lexica/japanese_politelex_purified.csv')
    zh_lex = pd.read_csv('../MLG/Lexica/chinese_politelex.csv')
    return en_lex, es_lex, ja_lex, zh_lex


def generate_lexica():
    
    en_lex, es_lex, ja_lex, zh_lex = load_lexica()
    en_strategies = construct_politeness_strategy_regexes(en_lex, "English")
    es_strategies = construct_politeness_strategy_regexes(es_lex, "Spanish")
    ja_strategies = construct_politeness_strategy_regexes(ja_lex, "Japanese")
    zh_strategies = construct_politeness_strategy_regexes(zh_lex, "Chinese")
    
    lexica_dict = {}
    lexica_dict["English"] = en_strategies
    lexica_dict["Chinese"] = zh_strategies
    lexica_dict["Japanese"] = ja_strategies
    lexica_dict["Spanish"] = es_strategies
    
    return lexica_dict