# Deep Neural Network for Image Classification

CS 4375.503

Danh Tran - dnt190001
Minh Nguyen - mnn200001

This is a deep learning project using a Deep Neural Network (DNN) to perform image classification on a dataset of images.

## Prerequisites

Before running the program, make sure you have Python 3 installed. You can check this by running the following command:

```bash
python3 --version
```

Next, make sure you have the following packages installed:

```bash
pip3 install numpy pandas opencv-python scikit-learn matplotlib
```

## Data Preparation

Place your dataset folder, named 'train-scene classification', inside the main project folder. The dataset folder should contain 'train.csv' and a folder named 'train' with all the image files.

## Running the Program

To run the program, execute the following command in the terminal:

```bash
python3 image_classification_dnn.py
```

## Results

The output consists of training and validation Root Mean Squared Error (RMSE) values for each epoch, the test loss, and test accuracy after training. Additionally, the program will log experiment results in a file named 'experiment_logs.log' and display a plot of the training and validation RMSE values over the epochs.

## Hyperparameters

You can customize the hyperparameters by adding new sets to the hyperparameter_sets list inside the main() function. Example:

```python
hyperparameter_sets = [
    {'learning_rate': 0.0001, 'hidden_layers': [128, 256, 512], 'batch_size': 32},
    {'learning_rate': 0.0005, 'hidden_layers': [64, 128, 256], 'batch_size': 64},
]
```

This will run the training and evaluation process for each set of hyperparameters, log the final training and validation RMSE values, and plot the validation RMSE values over the epochs.
#   S c e n e _ R e g c o n i t i o n 
 
 #   S c e n e _ R e g c o n i t i o n 
 
 