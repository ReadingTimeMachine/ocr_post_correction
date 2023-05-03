# ocr_post_correction

TO DO:
* notebooks codes
* vocabulary
* training weights recent
* code for original model where corrected model

retrain from scratch

environment: opence-v1.5.1

# The '0_exploration_new_data' Folder
Where you will load and organize your training data and validation data by sentences with source and target data. This data will be aligned for the model to train on. A vocabalary dictionary will be created to contain the characters that appear in the data. This notebook also performs Levenshtein metrics on data. 

# The '1_preprocessing_new_data' Folder
Uses train aligned and dev aligned and vocaublary to create train and dev tensors. Character to index (char2i) and Index to character (i2char) will also be created based on vocabulary. Contains code that generates data in batches for a large training set at the bottom.

# The '2_baseline_github_new_data' Folder
This is where you will train the model on your tensors. You can use the YourDataset class to work on creating a dataloader to handle large amounts of data. This notebook also contains modified code of the model's architecture. Under "model" is where you will train the data and run through trainining epochs to identify the best dev loss. Once the model is done training, you can use the matplotlib package to visulualize the training and validation loss

# The 'metrics_with saved checkpoint_new' folder
This where you can run the best model (where the dev loss is the lowest) on the testing data and perform an evaluation of character error rate (cer) before and after running the model.
