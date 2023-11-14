# CEO_reports_classification

## Overview

This repository contains code for training, storing, and using both TF-IDF and bag-of-words models to classify CEO reports. The main functionalities are separated into two blocks: Training and Predicting.

### Training Pipeline

1. **Prepare PDF Files for Training:**
   - Paste all PDF files for training into the `Training/Reports` folder.

2. **Fill Training Data:**
   - Fill in the `Training/Classification/train.xlsx` file with PDF file names, start and end pages.

3. **Extract Text:**
   - Extract text from PDF files and create `extract_text.csv` file in the `Training/Reports_Text` folder (set `extract_train` variable to True).

4. **Preprocess Text:**
   - Preprocess the text and create `processed_database.xlsx` file in the `Training/Reports_Text` folder (set `preprocess_train` variable to True).

5. **Train Models using TF-IDF:**
   - Create matrices in `Training/Reports_text/tf_idf` folder (set `create_matrices_train_tf` variable to True).
   - Train models (set `train_tf_idf` variable to True).

6. **Train Models using Bag of Words:**
   - Create train bag in `Training/Reports_text/bag` folder (set `create_train_bag` variable to True).
   - Train models (set `train_bag` variable to True).

7. **Parallel Training (Optional):**
   - To train both bag and TF-IDF models simultaneously, use the `fast_train` variable.

### Predicting Pipeline

1. **Prepare PDF Files for Prediction:**
   - Paste all PDF files for prediction into the `Prediction/Reports` folder.

2. **Extract Text for Prediction:**
   - Extract text from PDF files and create `extract_text.csv` file in the `Prediction` folder (set `extract_predict` variable to True).

3. **Preprocess Text for Prediction:**
   - Preprocess the text and create `processed_database.xlsx` file in the `Prediction` folder (set `preprocess_predict` variable to True).

4. **Predict using TF-IDF:**
   - Create `Prediction/prediction_tf.xlsx` and `Prediction/start_end.xlsx` files (set `predict_tf` variable to True).

5. **Predict using Bag of Words:**
   - Create train bag in `Prediction` folder (set `create_predict_bag` variable to True).
   - Create `Prediction/prediction_bag.xlsx` and `Prediction/start_end.xlsx` files (set `predict_bag` variable to True).

6. **Cut PDFs for Manual Search:**
   - If start pages are available in `start_end.xlsx`, use `cut_to_manual_search` variable to cut PDFs into 16-page files. Files will be written into `Prediction/CEO_Reports_Manual` folder.

7. **Cut PDFs into Final Reports:**
   - If end pages are available in `start_end.xlsx`, use `final_cut` variable to cut PDFs into final reports. Reports will be written into `Prediction/CEO_Reports_Final` folder.

8. **Retrain Neural Network:**
   - When start and end pages are available, use `create_retrain_file` to create a variable for retraining the neural network. Choose to retrain using TF-IDF or Bag depending on the model used for prediction (`retrain_nn_tf` or `retrain_nn_bag` variables).

## Technologies

- Python
- Scikit-learn
- Pandas
