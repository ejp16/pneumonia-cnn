# Convolutional neural network for pneumonia detection with tensorflow

This notebook implements a Convolutional Neural Network (CNN) to classify chest X-ray images as either "Normal" or "Pneumonia".

Main steps:

1. **Data Loading:** The dataset is loaded from three folders: train, test, and validation, using TensorFlow’s image_dataset_from_directory. The class names are extracted to map labels to class names.

2. **Preprocessing:** Images are resized to 160x160 pixels and converted to grayscale using OpenCV. The data is stored in NumPy arrays for further processing.

3. **Data Visualization:** The notebook visualizes the number of samples in each subset (train, validation, test) and the class distribution, showing that the dataset is unbalanced.

4. **Dataset Merging and Splitting:** All subsets are merged and then split into new training and validation sets using train_test_split from scikit-learn (80% training, 20% validation).

5. **Data Augmentation and Rescaling:** Data augmentation (random flips and rotations) and rescaling (normalizing pixel values) are applied to improve generalization and training efficiency.

6. **CNN Model Architecture:** A sequential CNN model is built with three convolutional layers, max pooling, a dense layer, dropout for regularization, and a softmax output for classification.

7. **Training:** The model is trained for 20 epochs, using the training and validation sets.

8. **Evaluation:** Training and validation accuracy and loss are plotted to evaluate the model’s performance and check for overfitting.

Link to the dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data
