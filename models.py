import pandas as pd
from sklearn import naive_bayes, svm
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import datetime
import numpy as np
import scipy.sparse as sp
import multiprocessing
from sklearn.ensemble import GradientBoostingClassifier
import pre_models
from analysis import save_prediction, check_performance

np.random.seed(500)


def train_naive(save_model: bool, save_data: bool, train_matrix, test_matrix, train_y, test_y, dataframe,
                bag_or_tf: str):
    """Train Naive model"""
    start_time = datetime.datetime.now()
    print(f'\nTraining Naive Start time: {start_time}')
    # fit the training dataset on the NB classifier
    naive = naive_bayes.MultinomialNB()
    naive.fit(train_matrix, train_y)
    # predict the labels on validation dataset
    predictions_nb = naive.predict(test_matrix)
    check_performance(test_y, predictions_nb, 'Naive', variables_str='', variables_dict={}, bag_or_tf=bag_or_tf)
    if save_model:
        if bag_or_tf == 'bag':
            dump(naive, 'Training/Models/bag/Naive.joblib')
        elif bag_or_tf == 'tf':
            dump(naive, 'Training/Models/tf_idf/Naive.joblib')
    if save_data:
        dataframe['naive_label'] = predictions_nb
    end_time = datetime.datetime.now()
    print(f'Training Naive End time: {end_time}')
    total_time = end_time - start_time
    print(f'Training Naive Total Time: {total_time}\n')


def train_svm(save_model: bool, save_data: bool, train_matrix, test_matrix, train_y, test_y, dataframe, bag_or_tf: str,
              kernel: str):
    """Train SVM model"""
    start_time = datetime.datetime.now()
    print(f'\nTraining SVM Start time: {start_time}')
    svm_m = svm.SVC(C=1.0, kernel=kernel, degree=3, gamma='auto', probability=True, decision_function_shape='ovr')
    svm_m.fit(train_matrix, train_y)
    predictions_svm = svm_m.predict(test_matrix)
    predictions_svm_prob = svm_m.predict_proba(test_matrix)
    if save_data:
        save_prediction(prediction=predictions_svm,
                        prediction_prob=predictions_svm_prob,
                        dataframe=dataframe,
                        model_name='svm')
    if save_model:
        if bag_or_tf == 'bag':
            dump(svm_m, 'Training/Models/bag/SVM.joblib')
        elif bag_or_tf == 'tf':
            dump(svm_m, 'Training/Models/tf_idf/SVM.joblib')
    variables_str = f'kernel = {kernel}'
    variables_dict = {'kernel': kernel}
    check_performance(test_y, predictions_svm, 'SVM', variables_str=variables_str, variables_dict=variables_dict,
                      bag_or_tf=bag_or_tf)
    end_time = datetime.datetime.now()
    print(f'Training SVM End time: {end_time}')
    total_time = end_time - start_time
    print(f'Training SVM Total Time: {total_time}\n')


def train_tree(save_model: bool, save_data: bool, train_matrix, test_matrix, train_y, test_y, dataframe, bag_or_tf: str,
               criterion: str):
    """Train Tree model"""
    start_time = datetime.datetime.now()
    print(f'\nTraining Tree Start time: {start_time}')
    tree_m = DecisionTreeClassifier(criterion=criterion, random_state=0)
    tree_m.fit(train_matrix, train_y)
    predictions_tree = tree_m.predict(test_matrix)
    predictions_tree_prob = tree_m.predict_proba(test_matrix)
    if save_data:
        save_prediction(prediction=predictions_tree,
                        prediction_prob=predictions_tree_prob,
                        dataframe=dataframe,
                        model_name='tree')
    if save_model:
        if bag_or_tf == 'bag':
            dump(tree_m, 'Training/Models/bag/Tree.joblib')
        elif bag_or_tf == 'tf':
            dump(tree_m, 'Training/Models/tf_idf/Tree.joblib')
    variables_str = f'kernel = {criterion}'
    variables_dict = {'criterion': criterion}
    check_performance(test_y, predictions_tree, 'Tree', variables_str=variables_str, variables_dict=variables_dict,
                      bag_or_tf=bag_or_tf)
    end_time = datetime.datetime.now()
    print(f'Training Tree End time: {end_time}')
    total_time = end_time - start_time
    print(f'Training Tree Total Time: {total_time}\n')


def train_extra_trees(save_model: bool, save_data: bool, train_matrix, test_matrix, train_y, test_y, dataframe,
                      bag_or_tf: str,
                      n_estimators: int,
                      max_depth,
                      min_samples_leaf: int):
    """Train Extra Tree model"""
    start_time = datetime.datetime.now()
    print(f'\nTraining Extra Trees Start time: {start_time}')
    extra_trees = ExtraTreesClassifier(random_state=0, n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_leaf=min_samples_leaf)
    extra_trees.fit(train_matrix, train_y)
    predictions_extra_trees = extra_trees.predict(test_matrix)
    predictions_extra_trees_prob = extra_trees.predict_proba(test_matrix)
    if save_data:
        save_prediction(prediction=predictions_extra_trees,
                        prediction_prob=predictions_extra_trees_prob,
                        dataframe=dataframe,
                        model_name='extra_tree')
    if save_model:
        if bag_or_tf == 'bag':
            dump(extra_trees, 'Training/Models/bag/Extra_Trees.joblib')
        elif bag_or_tf == 'tf':
            dump(extra_trees, 'Training/Models/tf_idf/Extra_Trees.joblib')
    variables_str = f'n_estimators = {n_estimators}; max_depth = {max_depth}; min_samples_leaf = {min_samples_leaf}'
    variables_dict = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf}
    check_performance(test_y, predictions_extra_trees, 'Extra_Trees',
                      variables_str=variables_str,
                      variables_dict=variables_dict, bag_or_tf=bag_or_tf)
    end_time = datetime.datetime.now()
    print(f'Training Extra Trees End time: {end_time}')
    total_time = end_time - start_time
    print(f'Training Extra Trees Total Time: {total_time}\n')


def train_random_forest(save_model: bool, save_data: bool, train_matrix, test_matrix, train_y, test_y, dataframe,
                        bag_or_tf: str,
                        n_estimators: int,
                        max_depth: int,
                        min_samples_leaf: int):
    """Train Random Forest model"""
    start_time = datetime.datetime.now()
    print(f'\nTraining Random Forest Start time: {start_time}')
    random_forest = RandomForestClassifier(random_state=0, n_estimators=n_estimators, max_depth=max_depth,
                                           min_samples_leaf=min_samples_leaf)
    random_forest.fit(train_matrix, train_y)
    predictions_random_forest = random_forest.predict(test_matrix)
    predictions_forest_prob = random_forest.predict_proba(test_matrix)
    if save_data:
        save_prediction(prediction=predictions_random_forest,
                        prediction_prob=predictions_forest_prob,
                        dataframe=dataframe,
                        model_name='random_forest')
    if save_model:
        if bag_or_tf == 'bag':
            dump(random_forest, 'Training/Models/bag/Random_Forest.joblib')
        elif bag_or_tf == 'tf':
            dump(random_forest, 'Training/Models/tf_idf/Random_Forest.joblib')
    variables_str = f'n_estimators = {n_estimators}; max_depth = {max_depth}; min_samples_leaf = {min_samples_leaf}'
    variables_dict = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf}
    check_performance(test_y, predictions_random_forest, 'Random_Forest',
                      variables_str=variables_str,
                      variables_dict=variables_dict, bag_or_tf=bag_or_tf)
    end_time = datetime.datetime.now()
    print(f'Training Random Forest End time: {end_time}')
    total_time = end_time - start_time
    print(f'Training Random Forest Total Time: {total_time}\n')


def train_nn(save_model, save_data, train_matrix, test_matrix, train_y, test_y, dataframe,
             bag_or_tf,
             activation,
             solver,
             learning_rate,
             max_iter,
             hidden_layer_sizes):
    """Train Neural Network model"""
    start_time = datetime.datetime.now()
    print(f'\nTraining NN Start time: {start_time}')
    nn = MLPClassifier(random_state=0,
                       max_iter=max_iter,
                       hidden_layer_sizes=hidden_layer_sizes,
                       solver=solver,
                       learning_rate=learning_rate,
                       activation=activation)
    nn.fit(train_matrix, train_y)
    predictions_nn = nn.predict(test_matrix)
    predictions_nn_prob = nn.predict_proba(test_matrix)
    if save_data:
        save_prediction(prediction=predictions_nn,
                        prediction_prob=predictions_nn_prob,
                        dataframe=dataframe,
                        model_name='nn')
    if save_model:
        if bag_or_tf == 'bag':
            dump(nn, 'Training/Models/bag/Neural_Network.joblib')
        elif bag_or_tf == 'tf':
            dump(nn, 'Training/Models/tf_idf/Neural_Network.joblib')
    variables_str = f'max_iter = {max_iter};' \
                    f' hidden_layer_sizes = {hidden_layer_sizes};' \
                    f' solver = {solver}; learning_rate = {learning_rate}'
    variables_dict = {'activation': activation,
                      'solver': solver,
                      'learning_rate': learning_rate,
                      'max_iter': max_iter,
                      'hidden_layer_sizes': hidden_layer_sizes}
    check_performance(test_y, predictions_nn, 'Neural_Network', variables_str=variables_str,
                      variables_dict=variables_dict, bag_or_tf=bag_or_tf)
    end_time = datetime.datetime.now()
    print(f'Training NN End time: {end_time}')
    total_time = end_time - start_time
    print(f'Training NN Total Time: {total_time}\n')


def train_grad_boost(save_model: bool, save_data: bool, train_matrix, test_matrix, train_y, test_y, dataframe,
                     bag_or_tf: str,
                     n_estimators: int,
                     max_depth: int,
                     min_samples_leaf: int):
    """Grad Boosting model"""
    start_time = datetime.datetime.now()
    print(f'\nTraining Grad Boost Start time: {start_time}')
    grad_boost = GradientBoostingClassifier(random_state=0, n_estimators=n_estimators, max_depth=max_depth,
                                            min_samples_leaf=min_samples_leaf)
    grad_boost.fit(train_matrix, train_y)
    predictions_grad_boost = grad_boost.predict(test_matrix)
    predictions_grad_boost_prob = grad_boost.predict_proba(test_matrix)
    if save_data:
        save_prediction(prediction=predictions_grad_boost,
                        prediction_prob=predictions_grad_boost_prob,
                        dataframe=dataframe,
                        model_name='grad_boost')
    if save_model:
        if bag_or_tf == 'bag':
            dump(grad_boost, 'Training/Models/bag/Grad_Boost.joblib')
        elif bag_or_tf == 'tf':
            dump(grad_boost, 'Training/Models/tf_idf/Grad_Boost.joblib')
    variables_str = f'n_estimators = {n_estimators}; max_depth = {max_depth}; min_samples_leaf = {min_samples_leaf}'
    variables_dict = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf}
    check_performance(test_y, predictions_grad_boost, 'Grad_Boost',
                      variables_str=variables_str,
                      variables_dict=variables_dict, bag_or_tf=bag_or_tf)
    end_time = datetime.datetime.now()
    print(f'Training Grad Boost End time: {end_time}')
    total_time = end_time - start_time
    print(f'Training Grad Boost Total Time: {total_time}\n')


def predict_svm(predict_matrix, dataframe, bag_or_tf: str = 'tf'):
    """Predict labels using SVM model"""
    start_time = datetime.datetime.now()
    print(f'\nPredicting SVM Start time: {start_time}')
    if bag_or_tf == 'bag':
        svm_m = load('Training/Models/bag/SVM.joblib')
    elif bag_or_tf == 'tf':
        svm_m = load('Training/Models/tf_idf/SVM.joblib')
    else:
        raise Exception(f'Wrong input of bar_or_tf variable: {bag_or_tf}')
    predictions_svm = svm_m.predict(predict_matrix)
    predictions_svm_prob = svm_m.predict_proba(predict_matrix)
    save_prediction(prediction=predictions_svm,
                    prediction_prob=predictions_svm_prob,
                    dataframe=dataframe,
                    model_name='svm')
    end_time = datetime.datetime.now()
    print(f'Predicting SVM End time: {end_time}')
    total_time = end_time - start_time
    print(f'Predicting SVM Total Time: {total_time}\n')


def predict_naive(predict_matrix, dataframe, bag_or_tf: str = 'tf'):
    """Predict labels using Naive model"""
    start_time = datetime.datetime.now()
    print(f'\nPredicting Naive Start time: {start_time}')
    if bag_or_tf == 'bag':
        naive = load('Training/Models/bag/Naive.joblib')
    elif bag_or_tf == 'tf':
        naive = load('Training/Models/tf_idf/Naive.joblib')
    else:
        raise Exception(f'Wrong input of bar_or_tf variable: {bag_or_tf}')
    predictions_naive = naive.predict(predict_matrix)
    dataframe['naive_label'] = predictions_naive
    end_time = datetime.datetime.now()
    print(f'Predicting Naive End time: {end_time}')
    total_time = end_time - start_time
    print(f'Predicting Naive Total Time: {total_time}\n')


def predict_tree(predict_matrix, dataframe, bag_or_tf: str = 'tf'):
    """Predict labels using Tree model"""
    start_time = datetime.datetime.now()
    print(f'\nPredicting Tree Start time: {start_time}')
    if bag_or_tf == 'bag':
        tree_m = load('Training/Models/bag/Tree.joblib')
    elif bag_or_tf == 'tf':
        tree_m = load('Training/Models/tf_idf/Tree.joblib')
    else:
        raise Exception(f'Wrong input of bar_or_tf variable: {bag_or_tf}')
    predictions_tree = tree_m.predict(predict_matrix)
    predictions_tree_prob = tree_m.predict_proba(predict_matrix)
    save_prediction(prediction=predictions_tree,
                    prediction_prob=predictions_tree_prob,
                    dataframe=dataframe,
                    model_name='tree')
    end_time = datetime.datetime.now()
    print(f'Predicting Tree End time: {end_time}')
    total_time = end_time - start_time
    print(f'Predicting Tree Total Time: {total_time}\n')


def predict_extra_trees(predict_matrix, dataframe, bag_or_tf: str = 'tf'):
    """Predict labels using Extra Trees model"""
    start_time = datetime.datetime.now()
    print(f'\nPredicting Extra Trees Start time: {start_time}')
    if bag_or_tf == 'bag':
        extra_trees = load('Training/Models/bag/Extra_Trees.joblib')
    elif bag_or_tf == 'tf':
        extra_trees = load('Training/Models/tf_idf/Extra_Trees.joblib')
    else:
        raise Exception(f'Wrong input of bar_or_tf variable: {bag_or_tf}')
    predictions_extra_trees = extra_trees.predict(predict_matrix)
    predictions_extra_trees_prob = extra_trees.predict_proba(predict_matrix)
    save_prediction(prediction=predictions_extra_trees,
                    prediction_prob=predictions_extra_trees_prob,
                    dataframe=dataframe,
                    model_name='extra_tree')
    end_time = datetime.datetime.now()
    print(f'Predicting Extra Trees End time: {end_time}')
    total_time = end_time - start_time
    print(f'Predicting Extra Trees Total Time: {total_time}\n')


def predict_random_forest(predict_matrix, dataframe, bag_or_tf: str = 'tf'):
    """Predict labels using Random Forest model"""
    start_time = datetime.datetime.now()
    print(f'\nPredicting Random Forest Start time: {start_time}')
    if bag_or_tf == 'bag':
        random_forest = load('Training/Models/bag/Random_Forest.joblib')
    elif bag_or_tf == 'tf':
        random_forest = load('Training/Models/tf_idf/Random_Forest.joblib')
    else:
        raise Exception(f'Wrong input of bar_or_tf variable: {bag_or_tf}')
    predictions_random_forest = random_forest.predict(predict_matrix)
    predictions_forest_prob = random_forest.predict_proba(predict_matrix)
    save_prediction(prediction=predictions_random_forest,
                    prediction_prob=predictions_forest_prob,
                    dataframe=dataframe,
                    model_name='random_forest')
    end_time = datetime.datetime.now()
    print(f'Predicting Random Forest End time: {end_time}')
    total_time = end_time - start_time
    print(f'Predicting Random Forest Total Time: {total_time}\n')


def predict_nn(predict_matrix, dataframe, bag_or_tf: str = 'tf'):
    """Predict labels using Neural Network model"""
    start_time = datetime.datetime.now()
    print(f'\nPredicting NN Start time: {start_time}')
    if bag_or_tf == 'bag':
        nn = load('Training/Models/bag/Neural_Network.joblib')
    elif bag_or_tf == 'tf':
        nn = load('Training/Models/tf_idf/Neural_Network.joblib')
    else:
        raise Exception(f'Wrong input of bar_or_tf variable: {bag_or_tf}')
    predictions_nn = nn.predict(predict_matrix)
    predictions_nn_prob = nn.predict_proba(predict_matrix)
    save_prediction(prediction=predictions_nn,
                    prediction_prob=predictions_nn_prob,
                    dataframe=dataframe,
                    model_name='nn')
    end_time = datetime.datetime.now()
    print(f'Predicting NN End time: {end_time}')
    total_time = end_time - start_time
    print(f'Predicting NN Total Time: {total_time}\n')


def predict_grad_boost(predict_matrix, dataframe, bag_or_tf: str = 'tf'):
    """Predict labels using Gradient Boost model"""
    start_time = datetime.datetime.now()
    print(f'\nPredicting Grad Boost Start time: {start_time}')
    if bag_or_tf == 'bag':
        grad_boost = load('Training/Models/bag/Grad_Boost.joblib')
    elif bag_or_tf == 'tf':
        grad_boost = load('Training/Models/tf_idf/Grad_Boost.joblib')
    else:
        raise Exception(f'Wrong input of bar_or_tf variable: {bag_or_tf}')
    predictions_grad_boost = grad_boost.predict(predict_matrix)
    predictions_grad_boost_prob = grad_boost.predict_proba(predict_matrix)
    save_prediction(prediction=predictions_grad_boost,
                    prediction_prob=predictions_grad_boost_prob,
                    dataframe=dataframe,
                    model_name='Grad_Boost')
    end_time = datetime.datetime.now()
    print(f'Predicting Grad Boost End time: {end_time}')
    total_time = end_time - start_time
    print(f'Predicting Grad Boost Total Time: {total_time}\n')


def train_models_bag():
    """Training all models using data from Bag of Words"""
    start_time = datetime.datetime.now()
    print(f'\nTraining Bag Start time: {start_time}')

    Train_Y = np.ravel(
        pd.read_csv('Training/Reports_text/bag/Train_Y.csv'))
    Test_Y = np.ravel(
        pd.read_csv('Training/Reports_text/bag/Test_Y.csv'))

    train_matrix = sp.load_npz('Training/Reports_text/bag/train_matrix.npz')
    test_matrix = sp.load_npz('Training/Reports_text/bag/test_matrix.npz')

    test_files_info = pd.DataFrame()

    pool = multiprocessing.Pool(2)

    # Call the functions concurrently using pool.apply_async
    pool.apply_async(train_svm, args=(True, True, train_matrix, test_matrix, Train_Y, Test_Y, test_files_info,
                                      'bag',
                                      'linear'))

    pool.apply_async(train_nn, args=(True, True, train_matrix, test_matrix, Train_Y, Test_Y, test_files_info,
                                     'bag',
                                     'logistic',
                                     'adam',
                                     'constant',
                                     900,
                                     200))
    # Close the pool
    pool.close()
    pool.join()

    # train_svm(train_matrix=train_matrix, test_matrix=test_matrix, train_y=Train_Y, test_y=Test_Y,
    #           dataframe=test_files_info,
    #           kernel='linear',
    #           save_model=True,
    #           save_data=True, bag_or_tf='bag')
    # train_nn(train_matrix=train_matrix, test_matrix=test_matrix, train_y=Train_Y, test_y=Test_Y,
    #          dataframe=test_files_info,
    #          activation='logistic',
    #          solver='adam',
    #          learning_rate='constant',
    #          max_iter=900,
    #          hidden_layer_sizes=200,
    #          save_model=True,
    #          save_data=True, bag_or_tf='bag')
    # train_extra_trees(train_matrix=train_matrix, test_matrix=test_matrix, train_y=Train_Y, test_y=Test_Y,
    #                   dataframe=test_files_info,
    #                   max_depth=None,
    #                   min_samples_leaf=1,
    #                   n_estimators=250,
    #                   save_model=True,
    #                   save_data=True, bag_or_tf='bag')
    # train_naive(train_matrix=train_matrix, test_matrix=test_matrix, train_y=Train_Y, test_y=Test_Y,
    #             dataframe=test_files_info,
    #             save_model=True,
    #             save_data=True, bag_or_tf='bag')
    # train_tree(train_matrix=train_matrix, test_matrix=test_matrix, train_y=Train_Y, test_y=Test_Y,
    #            dataframe=test_files_info,
    #            criterion='gini',
    #            save_model=True,
    #            save_data=True, bag_or_tf='bag')
    # train_random_forest(train_matrix=train_matrix, test_matrix=test_matrix, train_y=Train_Y, test_y=Test_Y,
    #                     dataframe=test_files_info,
    #                     max_depth=2,
    #                     min_samples_leaf=1,
    #                     n_estimators=10,
    #                     save_model=True,
    #                     save_data=True, bag_or_tf='bag')
    # train_grad_boost(train_matrix=train_matrix, test_matrix=test_matrix, train_y=Train_Y, test_y=Test_Y,
    #                  dataframe=test_files_info,
    #                  max_depth=2,
    #                  min_samples_leaf=1,
    #                  n_estimators=10,
    #                  save_model=True,
    #                  save_data=True, bag_or_tf='bag')

    end_time = datetime.datetime.now()
    print(f'Training Bag End time: {end_time}')
    total_time = end_time - start_time
    print(f'Training Bag Total Time: {total_time}\n')


def predict_data_bag():
    """Predicting labels using models trained using Bag of Words"""
    start_time = datetime.datetime.now()
    print(f'\nPredicting Bag Start time: {start_time}')

    matrix = sp.load_npz('Prediction/predict_matrix_bag.npz')
    data_to_predict = pd.read_excel(
        'Prediction/processed_database.xlsx',
        sheet_name='data')

    # Multiprocessing
    mgr = multiprocessing.Manager()
    shared_dataframe = mgr.dict()
    pool = multiprocessing.Pool(2)

    # Call the functions concurrently using pool.apply_async
    pool.apply_async(predict_svm, args=(matrix, shared_dataframe, 'bag'))

    pool.apply_async(predict_nn, args=(matrix, shared_dataframe, 'bag'))

    # Close the pool
    pool.close()
    pool.join()

    results_df = pd.DataFrame.from_dict(dict(shared_dataframe))

    del shared_dataframe, matrix

    with pd.ExcelWriter('Prediction/prediction_bag.xlsx') as writer:
        data_to_predict.to_excel(writer, sheet_name='data', index=False)
        results_df.to_excel(writer, sheet_name='data', startcol=data_to_predict.shape[1], index=False)

    end_time = datetime.datetime.now()
    print(f'Predicting Bag End time: {end_time}')
    total_time = end_time - start_time
    print(f'Predicting Bag Total Time: {total_time}\n')

    # predict_svm(predict_matrix=matrix, dataframe=data_to_predict, bag_or_tf='bag')
    # predict_nn(predict_matrix=matrix, dataframe=data_to_predict, bag_or_tf='bag')
    # predict_extra_trees(predict_matrix=matrix, dataframe=data_to_predict, bag_or_tf='bag')
    # predict_naive(predict_matrix=matrix, dataframe=data_to_predict, bag_or_tf='bag')
    # predict_tree(predict_matrix=matrix, dataframe=data_to_predict, bag_or_tf='bag')
    # predict_random_forest(predict_matrix=matrix, dataframe=data_to_predict, bag_or_tf='bag')
    # predict_grad_boost(predict_matrix=matrix, dataframe=data_to_predict, bag_or_tf='bag')


def train_models_tf_idf():
    """Training all models using data from TF-IDF"""
    start_time = datetime.datetime.now()
    print(f'\nTraining TF Start time: {start_time}')

    Train_Y = np.ravel(
        pd.read_csv('Training/Reports_text/tf_idf/Train_Y.csv'))
    Test_Y = np.ravel(
        pd.read_csv('Training/Reports_text/tf_idf/Test_Y.csv'))
    test_files_info = pd.read_csv('Training/Reports_text/tf_idf/test_files_info.csv')

    train_matrix = sp.load_npz('Training/Reports_text/tf_idf/train_matrix.npz')
    test_matrix = sp.load_npz('Training/Reports_text/tf_idf/test_matrix.npz')

    # Multiprocessing
    mgr = multiprocessing.Manager()
    shared_dataframe = mgr.dict()
    pool = multiprocessing.Pool(2)

    # Call the functions concurrently using pool.apply_async
    pool.apply_async(train_svm, args=(True, True, train_matrix, test_matrix, Train_Y, Test_Y, shared_dataframe,
                                      'tf',
                                      'linear'))

    pool.apply_async(train_nn, args=(True, True, train_matrix, test_matrix, Train_Y, Test_Y, shared_dataframe,
                                     'tf',
                                     'logistic',
                                     'adam',
                                     'constant',
                                     900,
                                     200))
    # Close the pool
    pool.close()
    pool.join()

    results_df = pd.DataFrame.from_dict(dict(shared_dataframe))
    del shared_dataframe
    del Train_Y, Test_Y
    del train_matrix, test_matrix

    with pd.ExcelWriter('Training/Reports_text/tf_idf/Test_Result.xlsx') as writer:
        test_files_info.to_excel(writer, sheet_name='data', index=False)
        results_df.to_excel(writer, sheet_name='data', startcol=test_files_info.shape[1], index=False)

    end_time = datetime.datetime.now()
    print(f'Training TF End time: {end_time}')
    total_time = end_time - start_time
    print(f'Training TF Total Time: {total_time}\n')

    # train_svm(train_matrix=train_matrix, test_matrix=test_matrix, train_y=Train_Y, test_y=Test_Y,
    #           dataframe=test_files_info,
    #           kernel='linear',
    #           save_model=True,
    #           save_data=True, bag_or_tf='tf')
    # train_nn(train_matrix=train_matrix, test_matrix=test_matrix, train_y=Train_Y, test_y=Test_Y,
    #          dataframe=test_files_info,
    #          activation='logistic',
    #          solver='adam',
    #          learning_rate='constant',
    #          max_iter=900,
    #          hidden_layer_sizes=200,
    #          save_model=True,
    #          save_data=True, bag_or_tf='tf')
    # train_extra_trees(train_matrix=train_matrix, test_matrix=test_matrix, train_y=Train_Y, test_y=Test_Y,
    #                   dataframe=test_files_info,
    #                   max_depth=None,
    #                   min_samples_leaf=1,
    #                   n_estimators=250,
    #                   save_model=True,
    #                   save_data=True, bag_or_tf='tf')
    # train_naive(train_matrix=train_matrix, test_matrix=test_matrix, train_y=Train_Y, test_y=Test_Y,
    #             dataframe=test_files_info,
    #             save_model=True,
    #             save_data=True, bag_or_tf='tf')
    # train_tree(train_matrix=train_matrix, test_matrix=test_matrix, train_y=Train_Y, test_y=Test_Y,
    #            dataframe=test_files_info,
    #            criterion='gini',
    #            save_model=True,
    #            save_data=True, bag_or_tf='tf')
    # train_random_forest(train_matrix=train_matrix, test_matrix=test_matrix, train_y=Train_Y, test_y=Test_Y,
    #                     dataframe=test_files_info,
    #                     max_depth=7,
    #                     min_samples_leaf=3,
    #                     n_estimators=150,
    #                     save_model=True,
    #                     save_data=True, bag_or_tf='tf')
    # train_grad_boost(train_matrix=train_matrix, test_matrix=test_matrix, train_y=Train_Y, test_y=Test_Y,
    #                  dataframe=test_files_info,
    #                  max_depth=7,
    #                  min_samples_leaf=3,
    #                  n_estimators=150,
    #                  save_model=True,
    #                  save_data=True, bag_or_tf='tf')


def predict_data_tf_idf():
    """Predicting labels using models trained using TF-IDF"""
    start_time = datetime.datetime.now()
    print(f'\nPredicting TF Start time: {start_time}')

    processed_data = pd.read_excel(
        'Training/Reports_text/processed_database.xlsx',
        sheet_name='data')

    data_to_predict = pd.read_excel(
        'Prediction/processed_database.xlsx',
        sheet_name='data')

    tfidf_data = pre_models.tf_idf(data_to_convert=data_to_predict['text'],
                                   data_frame_all_text=processed_data['text'])

    variables = data_to_predict.iloc[:, 3:]
    numeric_matrix = variables.astype(np.float64)
    csr_matrix = sp.csr_matrix(numeric_matrix)
    predict_matrix = sp.hstack([tfidf_data, csr_matrix])

    del tfidf_data, csr_matrix, numeric_matrix, variables, processed_data

    # Multiprocessing
    mgr = multiprocessing.Manager()
    shared_dataframe = mgr.dict()
    pool = multiprocessing.Pool(2)

    # Call the functions concurrently using pool.apply_async
    pool.apply_async(predict_svm, args=(predict_matrix, shared_dataframe, 'tf'))

    pool.apply_async(predict_nn, args=(predict_matrix, shared_dataframe, 'tf'))

    # Close the pool
    pool.close()
    pool.join()

    results_df = pd.DataFrame.from_dict(dict(shared_dataframe))
    del shared_dataframe

    with pd.ExcelWriter('Prediction/prediction_tf.xlsx') as writer:
        data_to_predict.to_excel(writer, sheet_name='data', index=False)
        results_df.to_excel(writer, sheet_name='data', startcol=data_to_predict.shape[1], index=False)

    end_time = datetime.datetime.now()
    print(f'Predicting TF End time: {end_time}')
    total_time = end_time - start_time
    print(f'Predicting TF Total Time: {total_time}\n')

    # predict_svm(predict_matrix=predict_matrix, dataframe=data_to_predict, bag_or_tf='tf')
    # predict_nn(predict_matrix=predict_matrix, dataframe=data_to_predict, bag_or_tf='tf')
    # predict_extra_trees(predict_matrix=predict_matrix, dataframe=data_to_predict, bag_or_tf='tf')
    # predict_naive(predict_matrix=predict_matrix, dataframe=data_to_predict, bag_or_tf='tf')
    # predict_tree(predict_matrix=predict_matrix, dataframe=data_to_predict, bag_or_tf='tf')
    # predict_random_forest(predict_matrix=predict_matrix, dataframe=data_to_predict, bag_or_tf='tf')
    # predict_grad_boost(predict_matrix=predict_matrix, dataframe=data_to_predict, bag_or_tf='tf')


def compare_models(bag_or_tf: str = 'tf'):
    """Training all models on different variables saving performance measures using multiprocessing"""
    df = pd.DataFrame()
    if bag_or_tf == 'bag':
        Train_Y = np.ravel(
            pd.read_csv('Training/Reports_text/bag/Train_Y.csv'))
        Test_Y = np.ravel(
            pd.read_csv('Training/Reports_text/bag/Test_Y.csv'))

        train_matrix = sp.load_npz('Training/Reports_text/tf_idf/train_matrix.npz')
        test_matrix = sp.load_npz('Training/Reports_text/tf_idf/test_matrix.npz')

        main_input = (False, False, train_matrix, test_matrix, Train_Y, Test_Y, df, 'bag')

    elif bag_or_tf == 'tf':
        Train_Y = np.ravel(
            pd.read_csv('Training/Reports_text/tf_idf/Train_Y.csv'))
        Test_Y = np.ravel(
            pd.read_csv('Training/Reports_text/tf_idf/Test_Y.csv'))

        train_matrix = sp.load_npz('Training/Reports_text/tf_idf/train_matrix.npz')
        test_matrix = sp.load_npz('Training/Reports_text/tf_idf/test_matrix.npz')

        main_input = (False, False, train_matrix, test_matrix, Train_Y, Test_Y, df, 'tf')

    else:
        raise Exception(f'Wrong input of bar_or_tf variable: {bag_or_tf}')

    variables = {
        'extra_trees': {
            'n_estimators': [250, 350, 450],
            'max_depth': [None],
            'min_samples_leaf': [1]
        },
        'nn': {
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'adam'],
            'learning_rate': ['constant'],
            'max_iter': [900, 1200],
            'hidden_layer_sizes': [200]
        },
        'svm': {
            'kernel': ['linear', 'sigmoid', 'rbf']
        },
        'tree': {
            'criterion': ['gini', 'log_loss', 'entropy']
        },
        'random_forest': {
            'n_estimators': [200, 300, 400],
            'max_depth': [15],
            'min_samples_leaf': [1]
        },
        'grad_boost': {
            'n_estimators': [200, 300, 400],
            'max_depth': [None, 3, 5, 15],
            'min_samples_leaf': [1, 2, 3]
        }
    }
    extra_tree_inputs = []
    for i, n_estimators in enumerate(variables['extra_trees']['n_estimators']):
        for n, max_depth in enumerate(variables['extra_trees']['max_depth']):
            for j, min_sample_leaf in enumerate(variables['extra_trees']['min_samples_leaf']):
                input = main_input + (n_estimators, max_depth, min_sample_leaf)
                extra_tree_inputs.append(input)

    if __name__ == "__main__":
        with multiprocessing.Pool(1) as p:
            start_time = datetime.datetime.now()
            print(f'\nExtra Trees Start time: {start_time}')
            p.starmap_async(train_extra_trees, extra_tree_inputs)
            p.close()
            p.join()
            end_time = datetime.datetime.now()
            print(f'Extra Trees End time: {end_time}')
            total_time = end_time - start_time
            print(f'Extra Trees Total Time: {total_time}\n')

    nn_inputs = []
    for i, activation in enumerate(variables['nn']['activation']):
        for n, solver in enumerate(variables['nn']['solver']):
            for j, learning_rate in enumerate(variables['nn']['learning_rate']):
                for k, max_iter in enumerate(variables['nn']['max_iter']):
                    for p, hidden_layer_sizes in enumerate(variables['nn']['hidden_layer_sizes']):
                        input = main_input + (activation, solver, learning_rate, max_iter, hidden_layer_sizes)
                        nn_inputs.append(input)

    if __name__ == "__main__":
        with multiprocessing.Pool(6) as p:
            start_time = datetime.datetime.now()
            print(f'\nNN Start time: {start_time}')
            p.starmap_async(train_nn, nn_inputs)
            p.close()
            p.join()
            end_time = datetime.datetime.now()
            print(f'NN End time: {end_time}')
            total_time = end_time - start_time
            print(f'NN Total Time: {total_time}\n')

    svm_inputs = []
    for i, kernel in enumerate(variables['svm']['kernel']):
        input = main_input + (kernel,)
        svm_inputs.append(input)

    if __name__ == "__main__":
        with multiprocessing.Pool(3) as p:
            start_time = datetime.datetime.now()
            print(f'\nSVM Start time: {start_time}')
            p.starmap_async(train_svm, svm_inputs)
            p.close()
            p.join()
            end_time = datetime.datetime.now()
            print(f'SVM End time: {end_time}')
            total_time = end_time - start_time
            print(f'SVM Total Time: {total_time}\n')

    tree_inputs = []
    for i, criterion in enumerate(variables['tree']['criterion']):
        input = main_input + (criterion,)
        tree_inputs.append(input)

    if __name__ == "__main__":
        with multiprocessing.Pool(3) as p:
            start_time = datetime.datetime.now()
            print(f'\nTree Start time: {start_time}')
            p.starmap_async(train_tree, tree_inputs)
            p.close()
            p.join()
            end_time = datetime.datetime.now()
            print(f'Tree End time: {end_time}')
            total_time = end_time - start_time
            print(f'Tree Total Time: {total_time}\n')

    random_forest_inputs = []
    for i, n_estimators in enumerate(variables['random_forest']['n_estimators']):
        for n, max_depth in enumerate(variables['random_forest']['max_depth']):
            for j, min_sample_leaf in enumerate(variables['random_forest']['min_samples_leaf']):
                input = main_input + (n_estimators, max_depth, min_sample_leaf)
                random_forest_inputs.append(input)

    if __name__ == "__main__":
        with multiprocessing.Pool(1) as p:
            start_time = datetime.datetime.now()
            print(f'\nRandom Forest Start time: {start_time}')
            p.starmap_async(train_random_forest, random_forest_inputs)
            p.close()
            p.join()
            end_time = datetime.datetime.now()
            print(f'Random Forest End time: {end_time}')
            total_time = end_time - start_time
            print(f'Random Forest Total Time: {total_time}\n')

    grad_boost_inputs = []
    for i, n_estimators in enumerate(variables['grad_boost']['n_estimators']):
        for n, max_depth in enumerate(variables['grad_boost']['max_depth']):
            for j, min_sample_leaf in enumerate(variables['grad_boost']['min_samples_leaf']):
                input = main_input + (n_estimators, max_depth, min_sample_leaf)
                grad_boost_inputs.append(input)

    if __name__ == "__main__":
        with multiprocessing.Pool() as p:
            start_time = datetime.datetime.now()
            print(f'\nGrad Boost Start time: {start_time}')
            p.starmap_async(train_grad_boost, grad_boost_inputs)
            p.close()
            p.join()
            end_time = datetime.datetime.now()
            print(f'Grad Boost End time: {end_time}')
            total_time = end_time - start_time
            print(f'Grad Boost Total Time: {total_time}\n')


def retrain_nn(bag_or_tf: str = 'tf', number_batches: int = 10):
    """Predicting labels using models trained using TF-IDF"""
    start_time = datetime.datetime.now()
    print(f'\nRetraining NN Start time: {start_time}')

    processed_data = pd.read_excel(
        'Training/Reports_text/processed_database.xlsx',
        sheet_name='data')

    data_to_retrain = pd.read_excel(
        'Prediction/retrain_data.xlsx',
        sheet_name='data')

    retrain_y = data_to_retrain['label']

    tfidf_data = pre_models.tf_idf(data_to_convert=data_to_retrain['text'],
                                   data_frame_all_text=processed_data['text'])

    del processed_data

    variables = data_to_retrain.iloc[:, 4:]
    numeric_matrix = variables.astype(np.float64)
    csr_matrix = sp.csr_matrix(numeric_matrix)
    retrain_matrix = sp.hstack([tfidf_data, csr_matrix])

    del tfidf_data, csr_matrix, numeric_matrix, variables

    if bag_or_tf == 'bag':
        nn = load('Training/Models/bag/Neural_Network.joblib')
    elif bag_or_tf == 'tf':
        nn = load('Training/Models/tf_idf/Neural_Network.joblib')
    else:
        raise Exception(f'Wrong input of bar_or_tf variable: {bag_or_tf}')

    for i in range(0, number_batches):
        x_1, x_2, y_1, y_2 = pre_models.sample_data(retrain_matrix, retrain_y, test_size=0.5)
        nn = nn.partial_fit(x_1, y_1)
        nn = nn.partial_fit(x_2, y_2)

    if bag_or_tf == 'bag':
        test_y = np.ravel(
            pd.read_csv('Training/Reports_text/bag/Test_Y.csv'))
        test_matrix = sp.load_npz('Training/Reports_text/bag/test_matrix.npz')
    elif bag_or_tf == 'tf':
        test_y = np.ravel(
            pd.read_csv('Training/Reports_text/tf_idf/Test_Y.csv'))
        test_matrix = sp.load_npz('Training/Reports_text/tf_idf/test_matrix.npz')
    else:
        raise Exception(f'Wrong input of bar_or_tf variable: {bag_or_tf}')

    predictions_nn = nn.predict(test_matrix)

    print(classification_report(y_true=test_y, y_pred=predictions_nn, zero_division="warn"))

    # if bag_or_tf == 'bag':
    #     dump(nn, 'Training/Models/bag/Neural_Network.joblib')
    # elif bag_or_tf == 'tf':
    #     dump(nn, 'Training/Models/tf_idf/Neural_Network.joblib')

    end_time = datetime.datetime.now()
    print(f'Retraining NN End time: {end_time}')
    total_time = end_time - start_time
    print(f'Retraining NN Total Time: {total_time}\n')


if __name__ == '__main__':
    retrain_nn('tf', 10)
