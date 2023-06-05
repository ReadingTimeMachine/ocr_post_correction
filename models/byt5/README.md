# Files for the byt5 Model

Files to train models:
 * `train_byt5_ocr_full.ipynb` (google colab, arXiv data)
 * `train_byt5_ocr_full_hal.ipynb` (training on [HAL](https://wiki.ncsa.illinois.edu/display/ISL20/HAL+cluster), arXiv data)
 * `train_byt5_ocr_full_hal_historical.ipynb` (training on [HAL](https://wiki.ncsa.illinois.edu/display/ISL20/HAL+cluster), historical data)
 
Inference:
 * `batch_inference_byt5.py` - run batch inference on a CPU in parallel with yt, arXiv data
 * `batch_inference_byt5_historical.py` - run batch inference on a CPU in parallel with yt, historical data
 * `inference_byte5_historical.ipynb` - test inference for historical documents

Calculations:
 * `baseline_byt5_calcs.ipynb` - various calculations with models

Pre-trained model weights:
 * See the `model_checkpoints.zip` file in the [Zenodo download]()
 
Training/validation/test data:
 * See the `historical_groundtruths.zip` in the [Zenodo download]() for pickle files of hand-annotated ground truths
 * See the `.csv` files in the [Zenodo download]() for sentences from the arXiv data