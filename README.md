# CNN_FACE-RECOGNITION 

# Facial Expression Recognition using EfficientNetB2

This README outlines the steps involved in the facial expression recognition project using a pre-trained EfficientNetB2 model.

## Project Setup

1. **Data Download:** The project begins by downloading a dataset (archive.zip) from Dropbox using `wget`.  This dataset contains training and testing images categorized by facial expressions.

2. **Unzip Data:** The downloaded archive is then unzipped using `unzip -q`. The `-q` flag suppresses verbose output.

3. **Import Libraries:** Import all the necessary libraries including TensorFlow, NumPy, Matplotlib, Pathlib, Scikit-learn, OpenCV, and Pickle.

4. **Path Definition:**  Define Pathlib objects for the training and testing image directories.

## Data Preparation

1. **Image and Label Loading:** The script loads the image paths and corresponding labels. Labels are extracted from the directory structure of the images.

2. **Label Encoding:** The labels (facial expressions) are encoded numerically using LabelEncoder from scikit-learn and then one-hot encoded using TensorFlow's `to_categorical`.

3. **Data Splitting:** The dataset is split into training and validation sets using `train_test_split` from scikit-learn.

4. **Class Weight Calculation:** Compute class weights to handle potential class imbalance in the training data. This assigns higher weights to less frequent classes during training.

## Data Augmentation and Preprocessing

1. **Data Loading Function:**  Defines a `load` function to read and decode JPEG images.

2. **Image Resizing:** A Keras `Sequential` model is created to resize images to a fixed size (IMG_SIZE x IMG_SIZE).

3. **Data Augmentation:** Further transformations are applied using data augmentation layers (random flips, rotations, and zoom) to increase the variability of the training data and enhance model robustness.

4. **Dataset Creation:** The `get_dataset` function creates TensorFlow datasets for training and validation. It applies the defined transformations, shuffling, and batching to the images and labels.  This function also includes the option to apply data augmentation during training.

5. **Dataset Verification:** The code verifies the shapes of the image and label tensors from the created datasets to check the correctness of the transformations. Sample images are visualized for verification.

## Model Training

1. **EfficientNetB2 Backbone:** A pre-trained EfficientNetB2 model (without the classification layers) is used as a feature extractor.

2. **Model Architecture:** Build the model by adding Global Average Pooling, Dropout, and dense layers on top of the EfficientNetB2 backbone. The output layer has a softmax activation for multi-class classification.

3. **Model Compilation:** The model is compiled with the Adam optimizer, categorical cross-entropy loss, and evaluation metrics (accuracy, precision, and recall).

4. **Training Phase 1:** The model is trained for a specified number of epochs, using the class weights computed earlier.

5. **Fine-tuning:** The base model (EfficientNetB2) is frozen, and training continues with a set of callbacks (ModelCheckpoint and EarlyStopping) to save the best model weights.

## Model Evaluation and Testing

1. **Test Dataset Preparation:** A dataset is prepared for testing using similar steps of preprocessing and loading as the training and validation sets.

2. **Model Evaluation:** The trained model is loaded with the best weights saved earlier and is evaluated on the test dataset to calculate loss, accuracy, precision, and recall.

3. **Model and Label Encoder Saving:** The trained model and LabelEncoder are saved for later use.

## Real-time Prediction

1. **Model and Label Encoder Loading (example):** The code includes placeholders for loading the saved model and label encoder, so they are ready for use.

2. **Image Preprocessing:**  A `process_image` function is defined to preprocess a single frame of video. This includes resizing and normalization.

3. **Camera Capture and Prediction:** The code captures video from a webcam, processes each frame, and makes a prediction using the loaded model. The predicted facial expression label is displayed on the video stream.

