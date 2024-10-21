# Deep-Learning-Projects
#Project Overview

This repository contains the implementation of a CNN-based model to classify images of cats and dogs. The model is trained on a dataset of cat and dog images and evaluated for its performance using various optimizers, including Adam, RMSprop, and SGD with momentum.

Files in the Repository:

#Q1.ipynb

Contains the code for calculating the total number of computations and parameters for the CNN model.

#Q2.ipynb

Implements the classification task for cat vs dog images with hyperparameter tuning, including optimizer choices like Adam, RMSprop, and SGD.
The code also integrates Weights & Biases (WandB) for tracking model training.

#Q3.ipynb

Builds a deeper CNN architecture to improve classification accuracy.
Applies enhanced image augmentation techniques for better generalization on unseen data.


#Dataset:
The dataset consists of images of cats and dogs, with the images split into training, validation, and test sets. The images are preprocessed using scaling and augmentation techniques such as rotation, zoom, and horizontal flipping to improve model robustness.


#Model Architecture:
A Convolutional Neural Network (CNN) model was used for this task, consisting of multiple layers of convolutional, max-pooling, and dropout layers to prevent overfitting. The final model is compiled with different optimizers, such as Adam, RMSprop, and SGD, and trained for 10 epochs with real-time logging via WandB.


#Dependencies:
      TensorFlow/Keras: Framework used for building and training the CNN model.
      WandB (Weights & Biases): Tool for experiment tracking and model performance monitoring.
      Matplotlib: For plotting accuracy and loss graphs.
      Pillow: Used for image processing.


Setup Instructions:
Clone this repository:
        
        git clone https://github.com/username/dog-vs-cat-classification.git
    cd dog-vs-cat-classification

Install the required dependencies:

    pip install -r requirements.txt
Set up your Weights & Biases API key:

    export WANDB_API_KEY=your_api_key
wandb login

Run the notebook of your choice:
                
                jupyter notebook Q2.ipynb

Training Details:
  
  Data Augmentation: Rotation, width and height shifts, zoom, and horizontal flips.
  Batch Size: 32
  Image Size: 64x64 pixels
  Number of Epochs: 10
  Optimizers: Adam, RMSprop, SGD, SGD with momentum

Results:
    Adam showed the best performance with a validation accuracy of approximately 90%, demonstrating fast convergence and stability.
    RMSprop and SGD with momentum also performed well, while SGD without momentum had slower convergence.


Model Evaluation:
    The best model achieved a test accuracy of approximately 90%.
    Several sample test images were classified correctly, with predictions visualized in the notebooks.

For more Details
                  
                  https://medium.com/@priyadharshaninavarathnam/dog-vs-cat-image-classification-using-cnns-c17f7ceeb9e5
