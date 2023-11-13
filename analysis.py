import pandas as pd
import datetime
import os
from itertools import islice
import math
import PyPDF2
from sklearn.metrics import classification_report


def save_prediction(prediction, prediction_prob, dataframe, model_name: str):
    """Saving predicted labels and probabilities to main DF"""
    prob_list = [prob for prob in prediction_prob]
    prob_0_list = [prob[0] for prob in prob_list]
    prob_1_list = [prob[1] for prob in prob_list]
    prob_2_list = [prob[2] for prob in prob_list]
    dataframe[f'{model_name}_label'] = prediction
    dataframe[f'{model_name}_prob_0'] = prob_0_list
    dataframe[f'{model_name}_prob_1'] = prob_1_list
    dataframe[f'{model_name}_prob_2'] = prob_2_list


def check_performance(test_y, prediction, model: str, variables_str: str, variables_dict: dict, bag_or_tf: str):
    """Calculating and saving performance measures of each model"""
    print(f'\nPerformance of the {model}')
    # printing and saving the report to txt
    print(f'{classification_report(y_true=test_y, y_pred=prediction)}\n')
    if bag_or_tf == 'bag':
        with open(f'Training/Models/bag/performance_{model}.txt', 'a') as f:
            f.writelines(f'\nPerformance of the {model} ({variables_str}) {bag_or_tf}'
                         f'\n{classification_report(y_true=test_y, y_pred=prediction, zero_division="warn")}\n')
    elif bag_or_tf == 'tf':
        with open(f'Training/Models/tf_idf/performance_{model}.txt', 'a') as f:
            f.writelines(f'\nPerformance of the {model} ({variables_str}) {bag_or_tf}'
                         f'\n{classification_report(y_true=test_y, y_pred=prediction, zero_division="warn")}\n')

    # saving the report to excel
    report = classification_report(y_true=test_y, y_pred=prediction, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    for var in variables_dict:
        df_report[var] = [variables_dict[var]] * 6
    labels = [0, 1, 2, 'accuracy', 'macro_avg', 'weighted_avg']
    df_report['labels'] = labels
    if bag_or_tf == 'bag':
        try:
            with pd.ExcelWriter(f'Training/Models/bag/performance_{model}.xlsx',
                                mode='a',
                                if_sheet_exists='overlay') as f:
                df_report.to_excel(f, sheet_name='data', header=False, index=False, startrow=f.sheets['data'].max_row)
        except FileNotFoundError:
            df_report.to_excel(f'Training/Models/bag/performance_{model}.xlsx',
                               sheet_name='data',
                               index=False,
                               header=True)
    elif bag_or_tf == 'tf':
        try:
            with pd.ExcelWriter(f'Training/Models/tf_idf/performance_{model}.xlsx',
                                mode='a',
                                if_sheet_exists='overlay') as f:
                df_report.to_excel(f, sheet_name='data', header=False, index=False, startrow=f.sheets['data'].max_row)
        except FileNotFoundError:
            df_report.to_excel(f'Training/Models/tf_idf/performance_{model}.xlsx',
                               sheet_name='data',
                               index=False,
                               header=True)


def post_prediction_analysis(bag_or_tf: str):
    """Creating one label from NN and SVM and creating file with start and end pages for each PDF file"""
    start_time = datetime.datetime.now()
    print(f'\nPrediction Analysis Start Time: {start_time}')

    if bag_or_tf == 'bag' or bag_or_tf == 'tf':
        prediction_df = pd.read_excel(f'Prediction/prediction_{bag_or_tf}.xlsx', sheet_name='data')
        prediction_df['syn_prob_0'] = prediction_df['svm_prob_0'] * prediction_df['nn_prob_0']
        prediction_df['syn_prob_1'] = prediction_df['svm_prob_1'] * prediction_df['nn_prob_1']
        prediction_df['syn_prob_2'] = prediction_df['svm_prob_2'] * prediction_df['nn_prob_2']
        max_indexes = prediction_df[['syn_prob_0', 'syn_prob_1', 'syn_prob_2']].idxmax(axis=1)
        prediction_df['syn_label'] = max_indexes.str.extract(r'(\d+)').astype(int)
        prediction_df.loc[:, ['file', 'code', 'page', 'syn_label', 'syn_prob']]\
            .to_excel(f'Prediction/prediction_{bag_or_tf}.xlsx', sheet_name='data', index=False)
    else:
        print('Please choose between bag and tf')
        return

    # Creating a file with pdf names, starting and ending pages
    file_names = set(prediction_df['file'])
    files_pages = []
    length_files = len(file_names)

    for index, file in enumerate(file_names):
        progress = (index + 1) / length_files * 100
        print(f"Progress Start_End file creation: {progress:.2f}%")

        if 1 in prediction_df[(prediction_df['file'] == file)]['syn_label'].values:
            max_prob_start = \
                prediction_df.loc[
                    (prediction_df['file'] == file) & (prediction_df['syn_label'] == 1), ['syn_prob']].max()[
                    0]
            start_page = \
                prediction_df.loc[
                    (prediction_df['file'] == file) & (prediction_df['syn_label'] == 1) & (
                            prediction_df['syn_prob'] == max_prob_start)][
                    'page'].iloc[0]
        else:
            start_page = ''

        if 2 in prediction_df[(prediction_df['file'] == file)]['syn_label'].values:
            max_prob_end = \
                prediction_df.loc[
                    (prediction_df['file'] == file) & (prediction_df['syn_label'] == 2), ['syn_prob']].max()[
                    0]
            end_page = \
                prediction_df.loc[
                    (prediction_df['file'] == file) & (prediction_df['syn_label'] == 2) & (
                            prediction_df['syn_prob'] == max_prob_end)][
                    'page'].iloc[0]
        else:
            end_page = ''

        if (start_page != '') and (end_page != '') and (start_page > end_page):
            end_page = ''

        files_pages.append({'file': file, 'start_page': start_page, 'end_page': end_page})

    del prediction_df

    pd.DataFrame(files_pages).to_excel('Prediction/start_end.xlsx', sheet_name='data', index=False)

    end_time = datetime.datetime.now()
    print(f'Prediction Analysis End time: {end_time}')
    total_time = end_time - start_time
    print(f'Prediction Analysis Total Time: {total_time}\n')


def prediction_reports_cut(stage: str):
    """Cutting PDF files to CEO reports"""
    start_time = datetime.datetime.now()
    print(f'\nCutting reports Start Time: {start_time}')
    start_end_df = pd.read_excel('Prediction/start_end.xlsx', sheet_name='data')
    length_file = len(start_end_df['file'])

    for index, row in islice(start_end_df.iterrows(), length_file):
        progress = (index + 1) / length_file * 100
        print(f"Progress Reports Cutting {stage}: {progress:.2f}%")

        file_name = str(row['file']) + '.pdf'

        if stage == 'Manual':
            input_path = f'Prediction/Reports/{file_name}'
            output_path = f'Prediction/CEO_Reports_Manual/{file_name}'
            if not os.path.isdir('Prediction/CEO_Reports_Manual'):
                os.makedirs('Prediction/CEO_Reports_Manual')
        elif stage == 'Final':
            input_path = f'Prediction/Reports/{file_name}'
            output_path = f'Prediction/CEO_Reports_Final/X_{file_name}'
            if not os.path.isdir('Prediction/CEO_Reports_Final'):
                os.makedirs('Prediction/CEO_Reports_Final')
        else:
            print('Wrong Stage please choose between Manual and Final')
            return

        if stage == 'Manual' and math.isnan(row['start_page']):
            with open('Prediction/error_files_cutting.txt', 'a') as file:
                file.write(f'{stage} {file_name}: no starting page\n')
            continue
        elif stage == 'Final' and (math.isnan(row['final_start_page']) or math.isnan(row['final_end_page'])):
            with open('Prediction/error_files_cutting.txt', 'a') as file:
                file.write(f'{stage} {file_name}: no starting or ending page\n')
            continue

        if stage == 'Manual':
            start_page = int(row['start_page'] - 1)
        elif stage == 'Final':
            start_page = int(row['final_start_page'] - 1)
        else:
            print('Wrong Stage please choose between Manual and Final')
            return

        if stage == 'Manual':
            end_page = int(row['start_page'] + 14)
        elif stage == 'Final':
            end_page = int(row['final_end_page'] - 1)
        else:
            print('Wrong Stage please choose between Manual and Final')
            return

        with open(input_path, 'rb') as input_file, open(output_path, 'wb') as output_file:
            try:
                pdf_reader = PyPDF2.PdfReader(input_file)
                pdf_writer = PyPDF2.PdfWriter()

                if len(pdf_reader.pages) < end_page:
                    end_page = len(pdf_reader.pages) - 1

                # Iterate through the specified pages and add them to the new PDF writer
                for page_number in range(start_page, end_page + 1):
                    page = pdf_reader.pages[page_number]
                    pdf_writer.add_page(page)

                # Save the modified PDF to the output file
                pdf_writer.write(output_file)
            except Exception as e:
                with open('Prediction/error_files_cutting.txt', 'a') as file:
                    file.write(f'{stage} {file_name}: {e}\n')

    end_time = datetime.datetime.now()
    print(f'Cutting reports End time: {end_time}')
    total_time = end_time - start_time
    print(f'Cutting reports Total Time: {total_time}\n')


def create_retrain_data():
    """Cutting PDF files to CEO reports"""
    start_time = datetime.datetime.now()
    print(f'\nCreating retrain data Start Time: {start_time}')

    # creating a set of names of files with CEO reports
    start_end_df = pd.read_excel('Prediction/start_end.xlsx', sheet_name='data')
    length_start_end = len(start_end_df)
    files_with_report = set()
    for index, row in enumerate(islice(start_end_df.iterrows(), length_start_end)):
        try:
            if not math.isnan(row[1]['final_start_page']) and \
                    not math.isnan(row[1]['final_end_page']) and \
                    row[1]['file'] != '':
                files_with_report.add(row[1]['file'])
        except TypeError as t:
            with open('Prediction/error_files_retraining.txt', 'a') as file:
                file.write(f' Wrong type in row in start_end: {index + 1}\n')

    print(files_with_report)

    # creating df with all data of files with ceo reports
    files_text = pd.read_excel('Prediction/processed_database.xlsx', sheet_name='data')
    files_text_filtered = files_text[files_text['file'].isin(files_with_report)]
    del files_text

    # creating a file with all variables needed to retrain the NN
    length_files_text = len(files_text_filtered)
    labels = []

    for index, row in enumerate(islice(files_text_filtered.iterrows(), length_files_text)):

        def find_start_end():
            start_end_row = start_end_df.loc[start_end_df['file'] == row[1]['file']]
            # if len(start_end_row) == 0 or math.isnan(start_end_row['final_start_page'].iloc[0]) or math.isnan(
            #         start_end_row['final_end_page'].iloc[0]):
            #     start = ''
            #     end = ''
            # else:
            start = int(start_end_row['final_start_page'].iloc[0])
            end = int(start_end_row['final_end_page'].iloc[0])
            return start, end

        if index == 0:
            start_page, end_page = find_start_end()
        elif index > 0 and row[1]['file'] != file_name:
            start_page, end_page = find_start_end()

        # if not str(start_page).isdigit() or not str(end_page).isdigit():
        #     labels.append(0)
        # else:
        if int(row[1]['page']) == start_page:
            labels.append(1)
        elif int(row[1]['page']) == end_page:
            labels.append(2)
        else:
            labels.append(0)

        file_name = row[1]['file']
        progress = (index + 1) / length_files_text * 100
        print(f"Progress retrain data: {progress:.2f}%")

    files_text_filtered.insert(loc=2,
                               column='label',
                               value=labels)

    files_text_filtered.to_excel('Prediction/retrain_data.xlsx', sheet_name='data', index=False)

    end_time = datetime.datetime.now()
    print(f'Creating retrain data End time: {end_time}')
    total_time = end_time - start_time
    print(f'Creating retrain data Total Time: {total_time}\n')
