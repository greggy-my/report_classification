import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
import datetime
import re
import unicodedata
from itertools import islice
from spellchecker import SpellChecker


def preprocess_extracted_text(path_to_data, encoding, sheet_name, excel):
    """Preprocessing of the extracted data (Stop-words, Token, Lemma)"""
    if excel:
        Corpus = pd.read_excel(path_to_data, sheet_name=sheet_name)
    else:
        Corpus = pd.read_csv(path_to_data, encoding=encoding)

    texts = list(Corpus['text'])

    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    pattern = r'[^a-zA-Z]'
    stopwords_set = set(stopwords.words('english'))
    word_Lemmatized = WordNetLemmatizer()

    text_after_token = []
    text_after_process = []
    spell = SpellChecker()
    pages_length = len(texts)

    for index, text in enumerate(islice(texts, pages_length)):
        replaced_text = re.sub(pattern, ' ', str(text))
        tokens = word_tokenize(replaced_text, 'english')
        text_after_token.append(tokens)

    del texts

    for index, tokens in enumerate(islice(text_after_token, pages_length)):
        progress = (index + 1) / pages_length * 100
        print(f"Progress in processing: {progress:.2f}%")
        new_tokens = []
        for token in islice(tokens, len(tokens)):
            new_token = unicodedata.normalize('NFKD', str(token).strip().encode('ASCII', 'ignore').decode()).lower()
            if new_token not in stopwords_set and len(new_token) > 1:
                # Turn on to improve grammar
                # if len(spell.unknown([new_token])) > 0:
                #     new_token = spell.correction(new_token)
                tag = pos_tag([new_token])
                new_tokens.append(word_Lemmatized.lemmatize(new_token, tag_map[tag[0][1]]))
                del tag
        text_after_process.append(new_tokens)

    del text_after_token

    Corpus['text'] = text_after_process
    Corpus['text'].dropna(inplace=True)
    return Corpus


def preprocess_train_data():
    """Preprocess and save the train data"""
    start_time = datetime.datetime.now()
    print(f'\nPreprocess train Start Time: {start_time}')
    Corpus = preprocess_extracted_text(
        path_to_data=r"Training/Reports_text/extracted_text_train.csv",
        encoding='latin-1', sheet_name='no', excel=False)

    Corpus.to_excel('Training/Reports_text/processed_database.xlsx',
                    sheet_name='data',
                    index=False)
    end_time = datetime.datetime.now()
    print(f'Preprocess train End time: {end_time}')
    total_time = end_time - start_time
    print(f'Preprocess train Total Time: {total_time}\n')


def preprocess_predict_data():
    """Preprocess and save the predict data"""
    start_time = datetime.datetime.now()
    print(f'\nPreprocess predict Start Time: {start_time}')
    Corpus = preprocess_extracted_text(
        path_to_data=r"Prediction/extracted_text_predict.csv",
        encoding='latin-1', sheet_name='no', excel=False)

    Corpus.to_excel('Prediction/processed_database.xlsx',
                    sheet_name='data',
                    index=False)
    end_time = datetime.datetime.now()
    print(f'Preprocess predict End time: {end_time}')
    total_time = end_time - start_time
    print(f'Preprocess predict Total Time: {total_time}\n')
