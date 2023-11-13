import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
import datetime
import numpy as np
import scipy.sparse as sp
from collections import Counter
from itertools import islice

np.random.seed(500)


def shuffle_rows(data_frame):
    data_frame = data_frame.sample(frac=1).reset_index(drop=True)
    return data_frame


def sample_data(variables, labels, test_size: float):
    """Sampling the data to Test and Train datasets"""
    train_x, test_x, train_y, test_y = model_selection.train_test_split(variables, labels,
                                                                        test_size=test_size)
    encoder = LabelEncoder()
    encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)
    print('Sampling - Done')
    return train_x, test_x, train_y, test_y


def tf_idf(data_to_convert, data_frame_all_text):
    """Converting tokens into numbers using TF-IDF"""
    tfidf_vect = TfidfVectorizer(max_features=25000)
    tfidf_vect.fit(data_frame_all_text)
    tfidf_data = tfidf_vect.transform(data_to_convert)
    print(f'TF-IDF - Done')
    return tfidf_data


def create_train_matrices_tf_idf():
    """Creating Test and Train data to train models using TF-IDF method"""
    start_time = datetime.datetime.now()
    print(f'\nCreating Matrices Start Time: {start_time}')

    Corpus = pd.read_excel('Training/Reports_text/processed_database.xlsx',
                           sheet_name='data')

    y = Corpus['label']

    Train_X, Test_X, Train_Y, Test_Y = sample_data(Corpus, y, test_size=0.1)
    Train_X_Tfidf = tf_idf(data_to_convert=Train_X['text'], data_frame_all_text=Corpus['text'])
    Test_X_Tfidf = tf_idf(data_to_convert=Test_X['text'], data_frame_all_text=Corpus['text'])

    pd.DataFrame(Train_Y).to_csv('Training/Reports_text/tf_idf/Train_Y.csv',
                                 index=False)
    pd.DataFrame(Test_Y).to_csv('Training/Reports_text/tf_idf/Test_Y.csv',
                                index=False)

    test_files_info = Test_X.iloc[:, 0:5]
    pd.DataFrame(test_files_info).to_csv('Training/Reports_text/tf_idf/test_files_info.csv',
                                         index=False)

    Train_X = Train_X.iloc[:, 4:]
    Test_X = Test_X.iloc[:, 4:]

    numeric_matrix_train = Train_X.astype(np.float64)
    csr_matrix_train = sp.csr_matrix(numeric_matrix_train)
    numeric_matrix_test = Test_X.astype(np.float64)
    csr_matrix_test = sp.csr_matrix(numeric_matrix_test)
    train_matrix = sp.hstack([Train_X_Tfidf, csr_matrix_train])
    test_matrix = sp.hstack([Test_X_Tfidf, csr_matrix_test])

    sp.save_npz('Training/Reports_text/tf_idf/train_matrix.npz', train_matrix)
    sp.save_npz('Training/Reports_text/tf_idf/test_matrix.npz', test_matrix)

    end_time = datetime.datetime.now()
    print(f'Creating Matrices End time: {end_time}')
    total_time = end_time - start_time
    print(f'Creating Matrices Total Time: {total_time}\n')


def create_bag(type_bag: str):
    """Creating matrix for Bag"""
    start_time = datetime.datetime.now()
    print(f'\nCreating Matrix {type_bag} Start time: {start_time}')

    def extract_data(corpus: pd.DataFrame):
        data = [s.strip('[]') for s in corpus['text']]
        data = [s.split(',') for s in data]
        pages = [[element.strip("'\" ") for element in sublist] for sublist in data]
        return pages

    if type_bag == 'train':
        Corpus = pd.read_excel('Training/Reports_text/processed_database.xlsx',
                               sheet_name='data')
        pages = extract_data(Corpus)
        set_words = set()
        all_words = []

        for index, page in enumerate(islice(pages, len(pages))):
            all_words += page
            print(f'Page {index}/{len(pages)} of building set of words')

        counter = Counter(all_words)

        for index, word in enumerate(islice(all_words, len(all_words))):
            print(f'Word {index}/{len(all_words)} of clearing words')
            if counter[word] > 25 and (word != '' and word != ' '):  # 25
                set_words.add(word)

        del counter
        del all_words

        pd.DataFrame({'set': list(set_words)}).to_csv('Training/Reports_text/bag/set_words.csv', index=False)

    elif type_bag == 'predict':
        Corpus = pd.read_excel(
            'Prediction/processed_database.xlsx',
            sheet_name='data')

        set_words = pd.read_csv('Training/Reports_text/bag/set_words.csv', header=None)
        set_words = set_words[0].values.tolist()
        pages = extract_data(Corpus)
    else:
        return print('Wrong type_bag choose from "train" and "predict"')

    NUMBER_DICTIONARIES = 10
    split_indexes = []
    for i in range(1, NUMBER_DICTIONARIES):
        if i == 1:
            split_index = len(set_words) // 10
            split_indexes.append(split_index)
        else:
            split_index = split_indexes[0] * i
            split_indexes.append(split_index)

    sets = []

    for i in range(0, NUMBER_DICTIONARIES):
        if i == 0:
            set1 = set_words[:split_indexes[0]]
            sets.append(set1)
        elif i == NUMBER_DICTIONARIES-1:
            set10 = set_words[split_indexes[8]:]
            sets.append(set10)
        else:
            set_i = set_words[split_indexes[i - 1]:split_indexes[i]]
            sets.append(set_i)

    dictionaries = []

    for i in range(0, NUMBER_DICTIONARIES):
        dictionaries.append({word: [] for word in islice(sets[i], len(sets[i]))})

    page_counters = [Counter(page) for page in islice(pages, len(pages))]
    matrices = []

    for i, dictionary in enumerate(islice(dictionaries, len(dictionaries))):
        for j, page_counter in enumerate(islice(page_counters, len(page_counters))):
            for word in islice(dictionary, len(dictionary)):
                dictionary[word].append(page_counter[word])
            progress = (j + 1) / len(page_counters) * 100
            print(f"Progress Dict_{i + 1}: {progress:.2f}%")
        word_dict_m = sp.csr_matrix(pd.DataFrame(dictionary).values)
        dictionary.clear()
        matrices.append(word_dict_m)
        print(f'Word Dict {i + 1}: {word_dict_m.shape}')

    del dictionaries
    del page_counters
    del pages

    if type_bag == 'train':
        extra_variables_matrix = sp.csr_matrix(Corpus.iloc[:, 4:].values)
        matrices.insert(0, extra_variables_matrix)
        matrix = sp.hstack(matrices)

        y = Corpus['label']

        train_x, test_x, train_y, test_y = sample_data(matrix, y, test_size=0.1)

        sp.save_npz('Training/Reports_text/bag/train_matrix.npz', train_x)
        sp.save_npz('Training/Reports_text/bag/test_matrix.npz', test_x)

        pd.DataFrame(test_y).to_csv('Training/Reports_text/bag/Test_Y.csv',
                                    index=False)
        pd.DataFrame(train_y).to_csv('Training/Reports_text/bag/Train_Y.csv',
                                     index=False)

    if type_bag == 'predict':
        extra_variables_matrix = sp.csr_matrix(Corpus.iloc[:, 3:].values)
        matrices.insert(0, extra_variables_matrix)

        matrix = sp.hstack(matrices)

        sp.save_npz('Prediction/predict_matrix_bag.npz', matrix)

    end_time = datetime.datetime.now()
    print(f'Creating Matrix {type_bag} End time: {end_time}')
    total_time = end_time - start_time
    print(f'Creating Matrix {type_bag} Total Time: {total_time}\n')
