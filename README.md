# Hyperspectral Data Generation and Unmixing with EGU-Net

## Description
This repository contains code for generating hyperspectral data and performing spectral unmixing using the EGU-Net model. The project is divided into two main Jupyter Notebooks:

1. **Data Generation (`map_gen.ipynb`)**: This notebook creates synthetic hyperspectral datasets.
2. **Unmixing and CNN Training (`EGU-Net.ipynb`)**: This notebook preprocesses the generated data, trains a Convolutional Neural Network (CNN) using the EGU-Net architecture, and performs predictions.


## Installation
Ensure you have Python installed. This project was tested with:
- Python: `3.9.18`

## Usage
### 1. Generate Hyperspectral Data
Run the `map_gen.ipynb` notebook to create a dataset. The dataset will be put in the ./data folder. This folder may need to be created before runing.

### 2. Perform Spectral Unmixing
Execute the `EGU-Net.ipynb` notebook to preprocess the data, train the CNN, and obtain predictions.

## Dependencies
The following Python libraries are required:
```
scipy==1.13.1
numpy==1.26.4
keras==2.10.0
matplotlib==3.9.2
scikit-learn==1.5.1
```

## Citation
If you use this work in your research, please cite the following paper:
[EGU-Net: A Deep Learning Approach for Hyperspectral Unmixing](https://example.com)  *(Replace with actual paper link)*

