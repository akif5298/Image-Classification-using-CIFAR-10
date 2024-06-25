# Image-Classification-using-CIFAR-10

**Project:** Mini Project 3  
**Course:** CP322 - Machine Learning   

---

## Introduction

This report presents a comparative analysis of two deep learning models, namely an Artificial Neural Network (ANN) and a Convolutional Neural Network (CNN), applied to the CIFAR-10 dataset. CIFAR-10 consists of 60,000 32x32 color images across 10 classes, making it a standard benchmark for image classification tasks. The objective of this study is to construct, train, and evaluate both models to classify CIFAR-10 images with high accuracy.

## Methodology

### Artificial Neural Network (ANN)

- **Architecture:** The ANN comprises multiple layers including an input layer, several hidden layers with 128 neurons each using ReLU activation, and an output layer with 10 neurons employing softmax activation.
  
- **Training Parameters:** Cross-entropy loss function, Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.01 and momentum of 0.9.

### Convolutional Neural Network (CNN)

- **Architecture:** The CNN includes two convolutional layers with ReLU activation followed by max-pooling layers. It consists of 16 and 32 filters in the respective convolutional layers, concluding with fully connected layers leading to 10 output neurons.
  
- **Training Parameters:** Similar to ANN, utilizing cross-entropy loss function and SGD optimizer with the same hyperparameters.

## Implementation

Both models were implemented using PyTorch, a versatile deep learning library. The CIFAR-10 dataset underwent standard preprocessing including normalization and conversion into tensors. The dataset was split into training and testing sets for effective model training and evaluation.

## Results

### ANN Results:


Certainly! Here's the complete README.md content in one block of Markdown:

markdown
Copy code
# Comparative Analysis of ANN and CNN Models on CIFAR-10 Dataset Report

**Project:** Mini Project 3  
**Course:** CP322 - Machine Learning  
**Submitted by:** Akif, Usama, Tharun, Peri, Mehti  
**Instructor:** Daruish Ibrahim  
**Due Date:** December 10, 2023  

---

## Introduction

This report presents a comparative analysis of two deep learning models, namely an Artificial Neural Network (ANN) and a Convolutional Neural Network (CNN), applied to the CIFAR-10 dataset. CIFAR-10 consists of 60,000 32x32 color images across 10 classes, making it a standard benchmark for image classification tasks. The objective of this study is to construct, train, and evaluate both models to classify CIFAR-10 images with high accuracy.

## Methodology

### Artificial Neural Network (ANN)

- **Architecture:** The ANN comprises multiple layers including an input layer, several hidden layers with 128 neurons each using ReLU activation, and an output layer with 10 neurons employing softmax activation.
  
- **Training Parameters:** Cross-entropy loss function, Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.01 and momentum of 0.9.

### Convolutional Neural Network (CNN)

- **Architecture:** The CNN includes two convolutional layers with ReLU activation followed by max-pooling layers. It consists of 16 and 32 filters in the respective convolutional layers, concluding with fully connected layers leading to 10 output neurons.
  
- **Training Parameters:** Similar to ANN, utilizing cross-entropy loss function and SGD optimizer with the same hyperparameters.

## Implementation

Both models were implemented using PyTorch, a versatile deep learning library. The CIFAR-10 dataset underwent standard preprocessing including normalization and conversion into tensors. The dataset was split into training and testing sets for effective model training and evaluation.

## Results

### ANN Results:

Epoch 1, Loss: 1.6638274238542523
Epoch 2, Loss: 1.4803416328052121
Epoch 3, Loss: 1.3915066207613787
Epoch 4, Loss: 1.3349204066464357
Epoch 5, Loss: 1.2865449158889253
Epoch 6, Loss: 1.2474587544455857
Epoch 7, Loss: 1.2107476574533127
Epoch 8, Loss: 1.176611542777942
Epoch 9, Loss: 1.1500847124687545
Epoch 10, Loss: 1.1191354838326155
Accuracy on the test set: 50.88%


### CNN Results:

Epoch 1, Loss: 1.6253152591798006
Epoch 2, Loss: 1.1869458079795399
Epoch 3, Loss: 0.9842461414654237
Epoch 4, Loss: 0.853221153549831
Epoch 5, Loss: 0.7415234768939445
Epoch 6, Loss: 0.6431446090683608
Epoch 7, Loss: 0.5566578101547782
Epoch 8, Loss: 0.47003701582665336
Epoch 9, Loss: 0.3908858009521156
Epoch 10, Loss: 0.32599819942241737
Accuracy on the test set: 68.8%


## Discussion

The CNN model outperforms the ANN in classifying CIFAR-10 images, achieving higher accuracy (68.8% vs. 50.88%). This superior performance is attributed to the CNN's ability to capture spatial hierarchies in images through its convolutional layers, which are crucial for accurate image recognition tasks. In contrast, the ANN, lacking this spatial awareness, shows limitations in handling complex image data.

## Conclusion

This study highlights the effectiveness of CNNs over ANNs for image classification tasks using the CIFAR-10 dataset. The CNN's architecture, designed for spatial understanding, proves more suitable for such tasks compared to the ANN's general-purpose structure. This project not only demonstrates practical application of deep learning concepts but also provides insights into model selection and performance evaluation in image recognition.
