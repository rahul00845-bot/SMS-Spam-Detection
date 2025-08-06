# SMS-Spam-Detection
Spam Message Detection using TensorFlow
=======================================

Overview:
---------
This project builds a machine learning model using TensorFlow to detect whether a given text message is spam or not. 
The model is trained on the popular SMS Spam Collection dataset.

Technologies Used:
------------------
- Python
- TensorFlow / Keras
- Pandas
- Matplotlib
- Scikit-learn

Dataset:
--------
Dataset used: spam.csv (SMS Spam Collection Dataset)

You can download the dataset from:
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Make sure the 'spam.csv' file is placed in the same directory as the Python script.

How to Run:
-----------
1. Make sure Python 3.x is installed on your system.

2. Install the required libraries:
   pip install pandas tensorflow matplotlib scikit-learn

3. Save the main script as `spam_classifier.py` and place it in the same folder as the `spam.csv` file.

4. Run the script from terminal:
   python spam_classifier.py

5. After training, the script will output final accuracy and display training vs validation graphs.

Features:
---------
- Text preprocessing using Tokenizer and Padding
- Label encoding for 'ham' and 'spam' messages
- Neural network with Embedding, Pooling, Dense layers
- Accuracy and loss visualizations using matplotlib

Output:
-------
- Console logs showing model training progress and final test accuracy
- Graphs for training vs validation accuracy and loss

Author:
-------
Rahul ladse
