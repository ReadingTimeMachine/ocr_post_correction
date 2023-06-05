# ocr_post_correction

Based on the open source model from: https://arxiv.org/abs/2109.06264 and [original GitHub repo](https://github.com/jarobyte91/post_ocr_correction).

As this model was not as accurate as the `byt5` model on our dataset, training data and model weights are not linked.

Files:
 * `0_exploration_new_data_new_pageLevel.ipynb` - Where you will load and organize your training data and validation data by sentences with source and target data. This data will be aligned for the model to train on. A vocabalary dictionary will be created to contain the characters that appear in the data. This notebook also performs Levenshtein metrics on data. (new pages has the most recent notebooks for the new data).
 
 * `1_preprocessing_new_data.ipynb` - Uses train aligned and dev aligned and vocaublary to create train and dev tensors. Character to index (char2i) and Index to character (i2char) will also be created based on vocabulary. Contains code that generates data in batches for a large training set at the bottom.

 * `2_baseline_github_new.ipynb` - This is where you will train the model on your tensors. This notebook also contains modified code of the model's architecture. Under "model" is where you will train the data and run through trainining epochs to identify the best dev loss. Once the model is done training, you can use the matplotlib package to visulualize the training and validation loss.  This file is for use on the [HAL cluster](https://wiki.ncsa.illinois.edu/display/ISL20/HAL+cluster) and uses the opence-v1.5.1 environment
 
 * `2_baseline_github_onColab.ipynb` is similar to `2_baseline_github_new.ipynb` but used for training on google colab
 
 * `batch_evaluate_windowed.py` does a batch evaulation using several windowing techniques with a trained model on a CPU cluster with yt.
 
 * `3_evaluate_new.ipynb` is setup for GPU evaluation in serial
 
 * `4_baseline_windowed_calcs.ipynb` provides evaluation calculations

