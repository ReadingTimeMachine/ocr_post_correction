# ocr_post_correction

This is the repository for TPDL 2023.  Organization is below:

## Files

* `explore_visualize_dataset.ipynb` explores character and word-level synthetic ground-truth (SGT) and OCR matches in several ways, including interactive Altair visualizations.
* `explore_aligned_dataset.ipynb` explores the aligned sentences used for training the models.  

## Data & Model Weights

520k (500k in training, 10k in validation, 10k in test) randomized, aligned sentences and model weights are found on the [Zenodo page for this work]() (embargo'd until publication).

## Folders

* `data/` - storage location of all data
  * `all_time_plot.csv` - data showing the arXiv Bulk Downloads and our dataset over our time range (1991-2011)
  * `letters.pickle` - Python dictionary with each SGT character as key, and all OCR-matched characters as values
  * `words.pickle` - Python dictionary with each SGT *word* as key, and all OCR-matched words as values
  * `words_cleaned.pickle` - Python dictionary with each SGT *word* as key, and all OCR-matched words as values.  
    Here, each SGT word has punctuation and captialization removed (this dictionary is smaller than the one in `words.pickle`)
* `models/`


# ------------------------

# TODO

So much stuff is getting updated right now!  More stuffs soon :D

TODO
* put in a fake tex document (already marked?)
* run toy model with TeXSoup, if not marked already
* run marking
* run alignment

* put in subset of randomized data with:
  1. aligned OCR sentence
  2. aligned PDF sentence
  3. PDF word type (not filled in)
  
* code for plots
