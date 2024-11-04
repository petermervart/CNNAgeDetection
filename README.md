# Age Prediction Based on Images with CNN

**Authors:** Michal Lüley, Peter Mervart

## Overview
This project aims to predict the age of individuals based on their images using a Convolutional Neural Network (CNN) implemented with TensorFlow. The model leverages a dataset sourced from Kaggle for training and evaluation.

## Dataset
- **Source:** [Age Prediction Dataset](https://www.kaggle.com/datasets/mariafrenti/age-prediction)
- The dataset consists of images categorized into 100 age groups. 

## Libraries Used
- TensorFlow
- Matplotlib
- WandB (Weights and Biases for tracking experiments)

## Methodology
1. **Data Loading:**
   - The dataset is split into training (90%), validation (5%), and test (5%) sets.
   - Images are loaded and resized to 128x128 pixels in grayscale.

2. **Data Cleaning:**
   - Removed around 1500 images that were incorrectly labeled or did not contain individuals.
   - Noted many mislabeled images, affecting the model's accuracy.

3. **Preprocessing and Augmentation:**
   - Experimented with various preprocessing methods, but found them not beneficial.

4. **Model Architecture:**
   - The CNN consists of 3 convolutional layers followed by pooling and dropout layers, leading to a final output layer with a single neuron to predict age as a continuous value.
   - Early stopping and checkpointing were implemented to avoid overfitting.

5. **Training:**
   - The model was trained with a batch size of 64 and a maximum of 1000 epochs, optimizing for Mean Absolute Error (MAE) as the loss metric.

6. **Evaluation:**
   - Achieved a Mean Absolute Error (MAE) of approximately 7.5 on the test set.
   - The model's performance was influenced by the complexity of age prediction based on appearance and the presence of mislabeled data.

## Results
The model demonstrates its ability to predict age from images, albeit with challenges due to mislabeled training data. The findings highlight the difficulties inherent in age prediction based on visual characteristics.
