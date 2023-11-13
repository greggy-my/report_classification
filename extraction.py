import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import datetime
import os
from itertools import islice
import fitz


def extract_train_text():
    """Extracting text from PDF files and assigning labels to each page from the classifier"""
    start_time = datetime.datetime.now()
    print(f'\nExtractor Start time: {start_time}')

    initial_data = pd.read_excel('Training/Classification/train.xlsx',
                                 sheet_name='Main sheet')
    labels = pd.DataFrame()
    labels['file'] = initial_data['file']
    labels['start_report'] = initial_data['start_report']
    labels['end_report'] = initial_data['end_report']
    labels = labels.dropna(subset=['start_report', 'end_report'], ignore_index=True)

    files_path = 'Training/Reports'
    files_to_check = os.listdir(files_path)

    writer = []

    error_file = open('Training/Reports_text/error_files_training.txt', 'a')

    # create file object variable
    # opening method will be rb
    for index, file in enumerate(files_to_check):
        print(f'Extracted Files: {index}/{len(files_to_check)}')

        file_name = file.replace('.pdf', '')
        file_row = labels.loc[labels['file'].isin([file_name])]
        code = file_name[:3]

        if len(file_row) != 0:
            start_page = int(file_row['start_report'].values[0])
            end_page = int(file_row['end_report'].values[0])

            try:
                doc = fitz.open(f'Training/Reports/{file}')
            except Exception as e:
                error_file.write(f"%s -> {e}\n" % file)
                continue

            doc_length = len(doc)
            for i, page in enumerate(islice(doc, doc_length)):
                number_of_page = i + 1

                try:
                    text = page.get_text()
                except Exception as e:
                    error_file.write(f"%s + {number_of_page} -> {e}\n" % file)
                    del text
                    continue

                if text != "":
                    if number_of_page == start_page:
                        label = 1
                    elif number_of_page == end_page:
                        label = 2
                    else:
                        label = 0

                    # save the extracted data from pdf
                    page_data_tuple = (file_name, code, text, label, number_of_page)
                    writer.append(page_data_tuple)
            doc.close()
    error_file.close()

    main_df = pd.DataFrame(writer, columns=['file', 'code', 'text', 'label', 'page'])
    del writer

    label_binarizer = LabelBinarizer()
    label_binarizer.fit(main_df['code'])
    output = label_binarizer.transform(main_df['code'])
    result_df = pd.DataFrame(output, columns=label_binarizer.classes_)
    main_df = pd.concat([main_df, result_df], axis=1)
    del result_df

    main_df.to_csv(
        'Training/Reports_text/extracted_text_train.csv',
        index=False)

    end_time = datetime.datetime.now()
    print(f'Extractor End time: {end_time}')
    total_time = end_time - start_time
    print(f'Extractor Total Time: {total_time}\n')


def extract_predict_text():
    """Extracting text from PDF files to predict labels"""
    start_time = datetime.datetime.now()
    print(f'\nExtractor Start time: {start_time}')

    files_path = 'Prediction/Reports'
    files_to_extract = os.listdir(files_path)

    writer = []

    error_file = open('Prediction/error_files_prediction.txt', 'a')

    for index, file in enumerate(files_to_extract):
        print(f'Extracted Files: {index}/{len(files_to_extract)}')

        file_name = file.replace('.pdf', '')
        code = file_name[:3]

        try:
            doc = fitz.open(f'Prediction/Reports/{file}')
        except Exception as e:
            error_file.write(f"%s -> {e}\n" % file)
            continue

        doc_length = len(doc)

        for i, page in enumerate(islice(doc, doc_length)):
            number_of_page = i + 1

            try:
                text = page.get_text()
            except Exception as e:
                error_file.write(f"%s + {number_of_page} -> {e}\n" % file)
                del text
                continue

            page_data_tuple = (file_name, code, text, number_of_page)
            writer.append(page_data_tuple)

            del text
            del page_data_tuple

        doc.close()

    error_file.close()

    main_df = pd.DataFrame(writer, columns=['file', 'code', 'text', 'page'])
    del writer
    label_binarizer = LabelBinarizer()
    codes_train = pd.read_csv(
        'Training/Reports_text/extracted_text_train.csv')
    label_binarizer.fit(codes_train['code'])
    output = label_binarizer.transform(main_df['code'])
    result_df = pd.DataFrame(output, columns=label_binarizer.classes_)
    main_df = pd.concat([main_df, result_df], axis=1)

    del result_df

    main_df.to_csv(
        'Prediction/extracted_text_predict.csv',
        index=False, encoding='latin-1', errors='replace')

    end_time = datetime.datetime.now()
    print(f'Extractor End time: {end_time}')
    total_time = end_time - start_time
    print(f'Extractor Total Time: {total_time}\n')



