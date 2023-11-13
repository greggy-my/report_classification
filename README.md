# CEO_reports_classification

Main principles:
1.	In order to use the code, you need to open only main.py file.
2.	All the functions are initialized using boolean variables in the main.py file. True – the function will be initialized, False – the function will not be initialized.
3.	All the functions don’t need any arguments to be inputted and can be used just by running main.py.
4.	Functions have an execution order because the previous function creates a data file that will be used by the next function. The functions are separated for convenience, since the execution time of one function with a large number of files can take several hours. 
5.	The code was written to train, store, and use both the TF-IDF and bag of words models. The data for training and the models themselves are stored in different folders.
6.	When you train models, past models are erased, and new ones are stored in their place.
7.	When you retrain models using the manually checked data from the prediction, models’ weights are updated and not erased.

Using functions:
1. There are 2 main blocks of function: Training and Predicting.
2. Training pipeline is: 
a.	Paste all PDF files for training to Training/Reports folder.
b.	Fill in Training/Classification/train.xlsx file with PDF file names, start and end pages.
c.	Extract text from PDF files and create extract_text.csv file in Training/Reports_Text folder (extract_train variable).
d.	Preprocess the text and create processed_database.xlsx file in Training/Reports_Text folder (preprocess_train variable).
e.	Train models using TF-IDF: 
i.	Create matrices in Training/Reports_text/tf_idf folder (create_matrices_train_tf variable)
ii.	Train models (train_tf_idf variable)
f.	Train models using Bag of Words:
i.	Create train bag in Training/Reports_text/bag folder (create_train_bag variable)
ii.	Train models (train_bag variable)
g.	If you would like to train both bag and tf-idf at the same time you can use fast_train variable which will parallel the training.
3. Predicting pipeline is: 
a.	Paste all PDF files for training to Prediction/Reports.
b.	Extract text from PDF files and create extract_text.csv file in Prediction folder (extract_predict variable).
c.	Preprocess the text and create processed_database.xlsx file in Prediction folder (preprocess_predict variable).
d.	Predict using TF-IDF: 
i.	Create Prediction/prediction_tf.xlsx and Prediction/start_end.xlsx files (predict_tf variable)
e.	Predicting using Bag of Words:
i.	Create train bag in Prediction folder (create_predict_bag variable)
ii.	Create Prediction/prediction_bag.xlsx and Prediction/start_end.xlsx files (predict_bag variable)
f.	When all the start pages will be in the start_end.xlsx use cut_to_manual_search variable to cut PDFs into small 16-page PDF files. All the small files will be written into Prediction/CEO_Reports_Manual folder.
g.	When all the end pages will be in the start_end.xlsx use final_cut variable to cut PDFs into the final reports. All the reports will be written into Prediction/CEO_Reports_Final folder.
h.	When there are all start and end pages for PDF files use create_retrain_file in order to create a variable to retrain the neural network. Then choose to retrain using TF-IDF or Bag depending on what model you used to predict (retrain_nn_tf or retrain_nn_bag variables).
