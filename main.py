import models
import multiprocessing
from analysis import post_prediction_analysis, prediction_reports_cut, create_retrain_data
from extraction import extract_train_text, extract_predict_text
from file_manager import unzip_arc
from pre_models import create_train_matrices_tf_idf, create_bag
from processing import preprocess_train_data, preprocess_predict_data


def main():
    # Extraction and preprocessing for training
    unzip = False
    extract_train = False
    preprocess_train = False

    # Creating data for TF and training models
    create_matrices_train_tf = False
    train_tf_idf = False

    # Creating data for Bag and training models
    create_train_bag = False
    train_bag = False

    # Extraction and preprocessing for prediction
    extract_predict = False
    preprocess_predict = False
    create_predict_bag = False

    # Prediction
    predict_bag = False
    predict_tf = False

    # Cutting PDF Files for Manual Check and Final Analysis
    cut_to_manual_search = False
    final_cut = False

    # Retraining NN on new manual data from prediction
    create_retrain_file = False
    retrain_nn_tf = True
    retrain_nn_bag = False

    # Fast train both Bag and TF (Only used with powerful CPU)
    fast_train = False

    # Launching the code
    if unzip:
        unzip_arc(path_to_save='All PDF Documents', path_to_unzip='All PDF Documents/Packages.zip')

    if extract_train:
        extract_train_text()

    if preprocess_train:
        preprocess_train_data()

    if create_matrices_train_tf:
        create_train_matrices_tf_idf()

    if create_train_bag:
        create_bag('train')

    if train_tf_idf:
        models.train_models_tf_idf()

    if train_bag:
        models.train_models_bag()

    if extract_predict:
        extract_predict_text()

    if preprocess_predict:
        preprocess_predict_data()

    if create_predict_bag:
        create_bag('predict')

    if predict_tf:
        models.predict_data_tf_idf()
        post_prediction_analysis('tf')

    if predict_bag:
        models.predict_data_bag()
        post_prediction_analysis('bag')

    if cut_to_manual_search:
        prediction_reports_cut('Manual')

    if final_cut:
        prediction_reports_cut('Final')

    if create_retrain_file:
        create_retrain_data()

    if retrain_nn_tf:
        models.retrain_nn('tf', 10)

    if retrain_nn_bag:
        models.retrain_nn('bag', 10)

    if fast_train:
        pool = multiprocessing.Pool()
        pool.apply_async(models.train_models_tf_idf, ())
        pool.apply_async(models.train_models_bag, ())
        pool.close()
        pool.join()


if __name__ == '__main__':
    main()
