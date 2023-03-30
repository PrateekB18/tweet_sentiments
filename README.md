# Sentiment Analysis of Tweets
This Jupyter notebook demonstrates the training and implementation of two powerful machine learning models, XGBoost Classifier and a Neural Network, for sentiment analysis of tweets. A labelled dataset of tweets was used for training both models. Subsequently, the trained models were used to predict the sentiment of an unlabelled dataset of tweets.

The notebook consists of the following sections:
<ul>
  <li>Importing necessary libraries and loading the labelled dataset</li>
  <li>Data preprocessing, including text cleaning and tokenization</li>
  <li>Visualising word cloud of the text in the tweets</li>
  <li>Hyperparameter tuning and training XGBoost classifier</li>
  <li>Defining and training a Tensorflow Sequential Model</li>
  <li>Using the trained models to predict the sentiments of unlabelled dataset of tweets</li>
</ul>

The hyperparameters of XGBoost were optimized using GridSearchCV. In order to train the TensorFlow model, a custom generator was implemented to feed batches of sparse data to the model. 
