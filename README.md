# OCT Scan Classification

A convolutional neural network (CNN) built with PyTorch to classify Optical Coherence Tomography (OCT) images into four diagnoses:  

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Status](https://img.shields.io/badge/Status-Complete-success)

## Diagnoses Classified:
* Choroidal Neovascularization (CNV)  
* Diabetic Macular Edema (DME)   
* Lipid and Protein Deposits (DRUSEN)  
* Normal Retina (NORMAL)  

## Features
- Custom CNN architecture built with PyTorch
- OCT retinal disease classification
- Image preprocessing and augmentation pipeline
- Validation accuracy tracking
- Automatic best-model checkpoint saving
- Confusion matrix visualization
- Classification report generation
- GPU acceleration with CUDA support
- Modular training and evaluation pipeline
- Separate saved-model evaluation workflow
- Makefile automation for training and evaluation

## Results

The CNN achieved strong classification performance on the OCT2017 test dataset.

| Metric | Score |
|---|---|
| Accuracy | 92% |
| Precision | 93% |
| Recall | 92% |
| F1-Score | 92% |


![Confusion Matrix](confusion_matrix.png)

## Exploratory Data Analysis

Exploratory data analysis (EDA) was performed to inspect:

- Class balance
- OCT image dimensions
- Pixel intensity distributions
- Sample retinal scan visualization

Additional analysis and visualizations are included in the notebook/ directory.

## Example OCT Scans
![image](https://github.com/user-attachments/assets/e26efce8-f5f8-4db3-bc47-fa0a241b21ff)

## Dataset
Data Source: https://www.kaggle.com/datasets/paultimothymooney/kermany2018/discussion/372147

Dataset splits:
- Train
- Validation
- Test

## Model Architecture
The CNN architecture consists of:

- 2 convolutional layers
- ReLU activation functions
- Max pooling layers
- Fully connected layers
- Dropout regularization

Input images are resized to 128x128 grayscale tensors before training.

## Disease Descriptions
### Choroidal Neovascularization (CNV)
The abnormal growth of new blood vessels within the choroid. 

### Diabetic Macular Edema (DME)  
Diabetes symptom that causes swelling in the macula, the central part of the retina responsible for sharp, detailed vision.   

### Lipid and Protein Deposits (DRUSEN)
Small, yellow deposits that accumulate under the retina in the macula.

## Installation

Clone the repository:

```bash
git clone https://github.com/LuccaDeFulgentis/OCT-scan-classification
cd OCT_scan_classification
```

Create a virtual environment:

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
make install
```

---

## Running the Project

Train the model:

```bash
make train
```

Evaluate the saved model:

```bash
make evaluate
```

## Sources

- [PyTorch CIFAR10 Tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

- [GeeksforGeeks CNN Overview](https://www.geeksforgeeks.org/convolutional-neural-network-cnn-in-machine-learning/)