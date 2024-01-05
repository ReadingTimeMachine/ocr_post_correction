# ocr_post_correction

This is the repository for TPDL 2023.  Organization is below:

## Files

* `explore_visualize_dataset.ipynb` explores character and word-level synthetic ground-truth (SGT) and OCR matches in several ways, including interactive Altair visualizations.
* `explore_aligned_dataset.ipynb` explores the aligned sentences used for training the models.  

## Data & Model Weights

520k (500k in training, 10k in validation, 10k in test) randomized, aligned sentences and model weights are found on the [Zenodo page for this work](https://zenodo.org/records/8006584).

## Folders

* `data/` - storage location of all data
  * `all_time_plot.csv` - data showing the arXiv Bulk Downloads and our dataset over our time range (1991-2011)
  * `letters.pickle` - Python dictionary with each SGT character as key, and all OCR-matched characters as values
  * `words.pickle` - Python dictionary with each SGT *word* as key, and all OCR-matched words as values
  * `words_cleaned.pickle` - Python dictionary with each SGT *word* as key, and all OCR-matched words as values.  
    Here, each SGT word has punctuation and captialization removed (this dictionary is smaller than the one in `words.pickle`)
* `models/`
  * `byt5/` contains all of the files needed to run and evaluate the byt5 model
  * `windowed/` contains all of the files needed to run and evaluate the windowed model
  * `mBART/` -- initial set up files for the `mBART`, but not used for the paper
* `example_alignment/` is an example of our alignment routine "in action".  Per our agreement with the arXiv, we cannot release our full dataset as of yet, but we hope this acts as an example of our methods.


# ------------------------

# TODO

1. add in HuggingFace links
