import zipfile
import datetime
import os
import shutil


def transfer_final_dataset():
    start_time = datetime.datetime.now()
    print(f'\nTransferring Start time: {start_time}')
    path_new_files = '/Users/grigorii/Library/CloudStorage/OneDrive-QueenMary,UniversityofLondon/Packages'
    set_new_files = set(os.listdir(path_new_files))
    print(len(set_new_files))
    path_train_files = 'Training/Reports'
    set_train_files = set(os.listdir(path_train_files))
    print(len(set_train_files))

    new_files_difference = [file for file in set.difference(set_new_files, set_train_files) if
                            file[len(file) - 3:] == 'pdf']

    print(new_files_difference)
    print(len(new_files_difference))

    length_unique_files = len(new_files_difference)
    error_files = []
    for index, file in enumerate(new_files_difference):
        print(f'Transfer Progress: {round(((index + 1) / length_unique_files), 4) * 100}% ')
        try:
            shutil.copyfile(f'{path_new_files}/{file}', f'Prediction/Reports/{file}')
        except:
            error_files.append(file)
            continue

    with open('Prediction/error_files_prediction.txt', 'w') as error_file:
        for file_name in error_files:
            error_file.write("%s\n" % file_name)

    print('Transferring: Done')
    end_time = datetime.datetime.now()
    print(f'Transferring End time: {end_time}')
    total_time = end_time - start_time
    print(f'Transferring Total Time: {total_time}\n')


def unzip_arc(path_to_unzip, path_to_save):
    """Unzip zip archives"""
    CHUNK_SIZE = 1024 * 1024 * 100  # 100 MB chunk size

    with zipfile.ZipFile(path_to_unzip, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            file_size = file_info.file_size

            # Extract the file in chunks
            with zip_ref.open(file_info) as file:
                extracted_file_path = path_to_save + '/' + file_info.filename

                os.makedirs(os.path.dirname(extracted_file_path), exist_ok=True)

                with open(extracted_file_path, 'wb') as output_file:
                    while True:
                        chunk = file.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        output_file.write(chunk)

        print('Unzipping complete!')
