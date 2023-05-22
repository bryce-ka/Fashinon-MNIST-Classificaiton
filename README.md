# Fashion MNIST Classifier

This repository contains code for loading and analyzing the Fashion MNIST dataset. It also includes an implementation of a dummy model and a random forest classifier for the classification task on this dataset.

## Dataset
The Fashion MNIST dataset is a collection of 60,000 28x28 grayscale images of 10 fashion categories, along with a test set of 10,000 images. The dataset is split into training and test sets, and each image is associated with a label indicating the corresponding fashion category.

## Requirements
The following packages are required to run the code:
- pandas
- numpy
- matplotlib
- scikit-learn

## Usage
To use this code, follow these steps:

1. Clone the repository: `git clone https://github.com/bryce-ka/Fashinon-MNIST-Classificaiton.git`
2. Navigate to the cloned repository: `cd Fashinon-MNIST-Classificaiton`
3. Ensure that the required packages are installed. You can install them using pip: `pip install -r requirements.txt`
4. Open the Jupyter Notebook `fashion_mnist.ipynb` using Jupyter or any other compatible notebook viewer.
5. Run the notebook cells to load and analyze the dataset, and to train and evaluate the classifiers.

## Code Overview
The `fashion_mnist.ipynb` notebook contains the following code sections:

1. **Data Loading**: This section includes functions for loading the Fashion MNIST dataset. The dataset is stored in gzip format, and the functions read and process the data to return the train and test images along with their labels.

2. **Data Analysis**: This section includes functions for analyzing the dataset. The `pretty_print` function is used to print a formatted representation of a single image from the dataset. The `plot_bar` function is used to plot a bar graph showing the distribution of clothing categories in the train and test sets.

3. **Dummy Model**: This section includes code for a dummy model that guesses the same clothing category for each sample of data. The accuracy of this model is 10% since there are 10 categories in the dataset.

4. **Random Forest Classifier**: This section includes code for training and evaluating a random forest classifier using Leave-One-Out Cross-Validation (LOOCV). The classifier is trained on different subsets of the training data, and its performance is evaluated on the test data. The average accuracy across all folds is calculated.

## Results
The dummy model achieves an accuracy of 10%, which is expected since it always predicts the same category. The random forest classifier achieves an accuracy of approximately 97.7%

Please refer to the notebook for more detailed code explanations and analysis.
