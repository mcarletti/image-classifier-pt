# Image Classifier in PyTorch

### Generate dataset

Load and save the dataset as binary files, one for training and one for test
```python
cd data
python3 generate_bin_data.py
```

### Train and Test custom model
Run the `train.sh` script from the root folder to train the model and `test.py` to test it. Run `python3 test.py -h` for a full list of testing parameters.
```python
./run_train.sh
python3 test.py --model_dir <path_to_model> --dataset_dir data/bin_data --use_gpu --verbose
```

### Test a VGG16 classifier
To run the out-of-the-box model based on VGG16, run `vgg16_classifier.py`
```python
python3 vgg16_classifier.py
```
