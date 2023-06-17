import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import csv
import fasttext
import fasttext.util
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator
# from translate import Translator
import argparse
import time, sys
import httpx
import re


# input: output language to be translated to
# output: dictionary of translations in the form {category: <translated_words>}
def translate_csv(source_lang, dest_lang, filename, output_filepath):  
    print("Translating LIWC...")  
    #read in pandas dataframe
    source_df = pd.read_csv(filename) 
    #get unique categories
    categories = np.unique(source_df["CATEGORY"])
    #get list of terms where category == category
    trans_categories = []
    trans_terms = []
    for category in tqdm(categories):
        terms = source_df.loc[source_df["CATEGORY"] == category]["word"]
        # translate terms in batches
        batch_size = 400
        for i in range(0, len(terms), batch_size):
            translate_string = ""
            for term in terms[i:min(i+batch_size, len(terms))]:
                translate_string += (str(term).lower().strip() + '\n')
            timeout = httpx.Timeout(5)
            translator = Translator(timeout=timeout)
            translator.raise_Exception = True
            # translator = Translator(to_lang=dest_lang_code)
            tries = 0
            while(tries < 3):
                try:
                    translated_output_string = translator.translate(translate_string, src=source_lang, dest=dest_lang).text
                    break
                    # translated_output_string = translator.translate(translate_string)
                except Exception as e:
                    translator = Translator()
                    translator.raise_Exception = True
                    tries += 1
                    print("tries: " + str(tries))
                    print("error: " + str(e))
                    continue
            translated_terms = translated_output_string.split('\n')
            for t in translated_terms:
                trans_categories.append(category)
                trans_terms.append(t)
               
    dest_df = pd.DataFrame(columns = ['CATEGORY', 'word'])
    dest_df["CATEGORY"] = trans_categories
    dest_df["word"] = trans_terms
    dest_df.to_csv(output_filepath, sep=',', encoding="utf-8", index=False)

# input: fasttext language
# output: fasttext model + dictionary of frequencies
def load_fasttext(language):
    if language == 'English':
        pre_trained_model = '/sandata/Fasttext/fastText/cc.en.300.bin'
    elif language == 'Hindi':
        pre_trained_model = '/sandata/Fasttext/fastText/cc.hi.300.bin'
    elif language == 'Chinese':
        pre_trained_model = '/sandata/Fasttext/fastText/cc.zh.300.bin'
    elif language == 'Spanish':
        pre_trained_model = '/sandata/Fasttext/fastText/cc.es.300.bin'
    elif language == 'Japanese':
        pre_trained_model = '/sandata/Fasttext/fastText/cc.ja.300.bin'
    elif language == 'Turkish':
        pre_trained_model = '/sandata/Fasttext/fastText/cc.tr.300.bin'
    elif language == 'Dutch':
        pre_trained_model = '/sandata/Fasttext/fastText/cc.nl.300.bin'
    else:
        raise ValueError('No pre-trained model for language: {}'.format(language))
    
    ft = fasttext.load_model(pre_trained_model)
    freqs = ft.get_words(include_freq=True, on_unicode_error='replace')
    print("loaded fasttext for: " + language)
    return ft, freqs

# input: list of seed words, full fasttext embeddings
# output: list of corresponding seed word embeddings
def get_embeddings(seed_words, embeddings_index):
    seed_embeddings = []
    for seed in seed_words:
        seed = re.sub(r'[^\w\s]', '', seed.lower().strip())
        seed = re.sub(" ", "-", seed)
        if seed not in embeddings_index: 
            print(seed + " not in embeddings")
        else:
            seed_emb = embeddings_index[seed]
            seed_embeddings.append(seed_emb)
    return seed_embeddings

# input: center embedding, output size, full fasttext embeddings
# output: dictionary of all words with cosine similarity to the center, descending order
def expand_embedding_vector(seed_center_embedding, output_size, embeddings_index):
    print("Full expansion...")
    full_expansion = []
    min_sim = 0
    all_words = list(embeddings_index.keys())
    for word in tqdm(all_words):
        if(word not in embeddings_index): continue
        try:
            cos_sim = cosine_similarity([seed_center_embedding],[embeddings_index[word]])[0][0]
        except:
            continue
        if cos_sim > min_sim:
            if len(full_expansion) == output_size:
                full_expansion = full_expansion[:-1]
            full_expansion.append((cos_sim, word))
            full_expansion.sort(key = lambda x: x[0], reverse = True)
            min_sim = full_expansion[-1][0]
    return full_expansion

# input: list of seed words, output size, full fasttext embeddings
# output: dictionary of all words with cosine similarity to the seed centroid, descending order
def full_expansion(seed_words, output_size, embeddings_index, freqs):
    print('found %s word vectors' % len(embeddings_index))
    seed_center_embedding = np.mean((get_embeddings(seed_words, embeddings_index)), axis=0)
    # print(seed_center_embedding)
    full_expansion = expand_embedding_vector(seed_center_embedding, output_size*5, embeddings_index)
    full_expansion = purify_list(full_expansion, freqs, output_size)
    return full_expansion

# input: list of seed words, output size, fasttext model, list of frequencies
# output: dictionary of all words with cosine similarity to each individual word, descending order
def individual_expansion(seed_words, output_size, ft_model, freqs):
    print("Individual expansion...")
    indv_expansion = {}
    for seed_word in tqdm(seed_words):
        # process word
        seed_word = re.sub(r'[^\w\s]', '', seed_word.lower().strip())
        seed_word = re.sub(" ", "-", seed_word)
        seed_word = seed_word.encode('utf-8')
        #get nearest neighbors
        try:
            closest_k = ft_model.get_nearest_neighbors(seed_word, k=output_size*5)
        except:
            continue
        closest_k = purify_list(closest_k, freqs, output_size)
        indv_expansion[seed_word] = closest_k
    return indv_expansion

#input: list of words to purify, list of fasttext frequencies
#output: purified list with low frequency words removed
def purify_list(nearest_words, freqs, output_size, direct_trans=False):
    threshold = np.percentile(freqs[1], 50)
    freqs_index = dict(zip(freqs[0], freqs[1]))
    purified_words = []
    added_words = []

    for word in nearest_words: 
        processed_word = (word[1].lower().split(".")[0]).replace("-", " ")
        try:
            freq = freqs_index[(processed_word)]
        except:
            freq = 0
        if(freq > threshold and processed_word not in added_words):
            purified_words.append((word[0], processed_word))
            added_words.append(processed_word)
        elif(direct_trans and ("-" in word)):
            purified_words.append((word[0], processed_word))
            added_words.append(processed_word)
    return purified_words[:output_size]

#input: list of tuples in the form (distance, word), where distance is cos similarity + threshold
def process_expanded_list(expanded_list, threshold):
    #remove all words that have length < 2
    expanded_list = [word for word in expanded_list if len(word[1]) > 1]
    #remove all elements of tuple where first value is < 0.7
    expanded_list = [word for word in expanded_list if word[0] > threshold]
    #sort list by first value in tuple
    expanded_list.sort(key = lambda x: x[0], reverse = True)
    #return sorted list
    return expanded_list

#input: name of language, language code, whether to translate before expansion
def expand_politelex(dest_lang: str, dest_lang_code: str, filename):
    #read in translated file and create dictionary of {category: words}
    translated_politelex_df = pd.read_csv(filename)
    ft_model, freqs  = load_fasttext(dest_lang)
    df_terms = []
    df_distances = []
    df_categories = []

    words = ft_model.get_words(on_unicode_error='replace')
    print("Loading word embeddings...")
    embeddings_index = {}
    for w in tqdm(words):
        embeddings_index[w] = ft_model.get_word_vector(w)

    #individual expansion
    terms = np.unique(translated_politelex_df["word"].to_list())
    expanded_terms_indv = individual_expansion(terms, 5, ft_model, freqs)

    #categorical expansion
    categories = np.unique(translated_politelex_df["CATEGORY"])
    for category in categories:
        print("category: " + category + "\n")
        input_seed_words = translated_politelex_df.loc[translated_politelex_df["CATEGORY"] == category]["word"]
        expanded_terms_concept = full_expansion(input_seed_words, 100, embeddings_index, freqs)

        additional_terms_indv = []
        #add relevant individual expansion words
        for seed in input_seed_words:
            seed = re.sub(r'[^\w\s]', '', seed.lower().strip())
            seed = re.sub(" ", "-", seed)
            seed = seed.encode('utf-8')
            try:
                indv_words = expanded_terms_indv[seed]
            except:
                continue
            for word in indv_words:
                additional_terms_indv.append(word)
        additional_terms_indv = process_expanded_list(additional_terms_indv, 0.7)
        print(additional_terms_indv)

        #add relevant categorical expansion words
        additional_terms_concept = []
        for word in expanded_terms_concept:
            additional_terms_concept.append(word)
        additional_terms_concept = process_expanded_list(additional_terms_concept, 0.5)
        print(additional_terms_concept)
        
        #add relevant seed words
        original_seed_words = []
        for word in input_seed_words:
            word = re.sub(r'[^\w\s]', '', word.lower().strip())
            word = re.sub(" ", "-", word)
            original_seed_words.append((1, word))
        original_seed_words = purify_list(original_seed_words, freqs, len(original_seed_words), direct_trans=True)

        #combine three expansion methods
        additional_terms = []
        for word in additional_terms_indv:
            additional_terms.append(word)
        for word in additional_terms_concept:
            additional_terms.append(word)
        for word in original_seed_words:
            additional_terms.append(word)
        additional_terms.sort(key = lambda x: x[0], reverse = True)

        
        for word in additional_terms:
            df_distances.append(word[0])
            df_terms.append(word[1])
            df_categories.append(category)

    politelex_expanded_df = pd.DataFrame(columns = ["CATEGORY", "word", "distance"])
    politelex_expanded_df["CATEGORY"] = df_categories
    politelex_expanded_df["word"] = df_terms
    politelex_expanded_df["distance"] = df_distances
    outfile = dest_lang.lower() + "_politelex_expanded" + ".csv"
    politelex_expanded_df.to_csv(outfile, sep=',', encoding="utf-8", index=False)

#main function, read in command line parameters
if __name__ == '__main__':
    #read in following parameters from command line: input, output, source, dest
    parser = argparse.ArgumentParser(description='Expand PoliteLex')
    parser.add_argument('--input', type=str, help='input file path', default="LIWC2015_purified_full.csv")
    parser.add_argument('--dest', type=str, help='destination language', choices=["English", "Hindi", "Chinese", "Spanish", "Japanese", "Turkish", "Dutch"])
    parser.add_argument("--dest_code", type=str, help='destination language code', choices=["en", "hi", "zh", "es", "ja", "tr", "nl"])

    args = parser.parse_args()

    #check to make sure dest args are not None
    if(args.dest is None or args.dest_code is None or args.input is None):
        raise ValueError("input file, destination language, and code must be specified")

    expand_politelex(args.dest, args.dest_code, args.input)  