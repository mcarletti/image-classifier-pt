# Image Classifier in PyTorch

### Generate dataset

Load and save the dataset as binary files, one for training and one for test
```python
cd data
python3 generate_bin_data.py
```

### Train and Test custom model
Run the `train.py` script from the root folder to train the model and `test.py` to test it
```python
python3 train.py  # train [80% acc]
python3 test.py   # test  [78% acc]
```

### Test a VGG16 classifier
To run the out-of-the-box model based on VGG16, run `vgg16_classifier.py`
```python
python3 vgg16_classifier.py
```
