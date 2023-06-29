# Pulp Exposure Detection Algorithm

This repository contains the implementation of an algorithm for the detection of tooth pulp exposure from bitewing X-ray
images. The algorithm is implemented using Python, PyTorch and PyTorch Lightning.

## Prerequisites

Before you begin, ensure you have met the following requirements:

You have a machine with Python 3.9+ installed.

You have installed the dependencies listed in requirements.txt.

## Data Setup

To use this algorithm, you need to set up the data as follows:

1. Place a NumPy array file named data.npy in the data/final directory. Each row in this array represents a sample. The
   columns in this file represent the following:

    - Column 1: The directory name where the corresponding image is stored.
    - Column 2: The ground truth for pulp exposure for the sample.
    - Column 3: The type of treatment (e.g., complete excavation, stepwise excavation).
    - Column 4: The distance between the lesion and the pulp.

2. In the same data directory, store the corresponding images in directories named according to the names in the
   data.npy file. Each image should be a crop of the tooth and can have varying sizes.

# Usage Instructions

Follow these steps to train and test your own model:

1. Perform a train-test-validation split by running the following command:

```
python src/data/scripts/train_val_test_split.py
```

2. Train and test your model:

```
python src/data/model/resnet/scripts/train.py
```