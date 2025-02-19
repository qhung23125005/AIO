# Convolutional Neural Networks (CNNs)

## Introduction
Convolutional Neural Networks (CNNs) are a class of deep learning models specifically designed to process structured grid data, such as images and text sequences. CNNs are widely used in computer vision tasks such as image classification, object detection, and segmentation, but they have also found applications in natural language processing and other fields.

## Key Components of CNNs

### 1. **Convolutional Layer**
   - Applies convolutional filters (kernels) to extract spatial features from input data.
   - Uses a set of small trainable filters that slide over the input to detect patterns.
   - Output is known as a feature map.

### 2. **Pooling Layer**
   - Reduces the dimensionality of feature maps while preserving key features.
   - Common types:
     - **Max Pooling**: Selects the maximum value in a window.
     - **Average Pooling**: Computes the average value in a window.

### 3. **Fully Connected Layer**
   - Flattens the feature maps and passes them to a fully connected (dense) layer.
   - Used for final classification or regression tasks.

## Architecture of a CNN
A basic CNN model typically consists of:
1. Input layer (image or sequence data)
2. Multiple convolutional layers with activation functions (e.g., ReLU)
3. Pooling layers to downsample feature maps
4. Fully connected layers for decision-making
5. Softmax or sigmoid layer for classification output

## CNN Workflow Example
1. **Input:** Image (e.g., 28x28 grayscale image of a digit)
2. **Convolution:** Apply multiple filters to detect edges, textures, and patterns.
3. **Activation (ReLU):** Introduce non-linearity to improve learning.
4. **Pooling (Max Pooling):** Reduce feature map size.
5. **Repeat:** Apply more convolution, activation, and pooling layers.
6. **Flatten:** Convert feature maps into a 1D vector.
7. **Fully Connected Layer:** Make predictions based on extracted features.
8. **Output:** Classification scores (e.g., digit 0-9 for handwritten digit recognition).

## Advantages of CNNs
- Automatically extract features from data.
- Reduce the number of parameters compared to fully connected networks.
- Improve generalization and reduce overfitting.
- Effective in spatially correlated data (e.g., images, speech, text).

## Exercise
In this week exercise, I will apply CNN for a number of classification tasks. Additionally, I will also try to use streamlit

- MNIST: https://github.com/qhung23125005/CNN-Applications---MNIST
- Cassava Leaf Disease: https://github.com/qhung23125005/CNN-Applications---Cassava-Leaf-Disease
- Sentiment Analysis: https://github.com/qhung23125005/CNN-Applications---Sentiment-Analysis
