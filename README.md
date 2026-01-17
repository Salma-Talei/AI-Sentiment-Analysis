# AI-Sentiment-Analysis
A project combining handwritten digit recognition (CNN) and sentiment analysis on Facebook data (NLP).
This project demonstrates the integration of two distinct Artificial Intelligence solutions into a single Python environment: Computer Vision for digit recognition and Natural Language Processing (NLP) for emotion detection.
## Projects Overview
### 1. Computer Vision (CNN)
A Convolutional Neural Network (CNN) trained to recognize handwritten digits (0-9).

  #### Architecture:
   7 layers including Convolutional (Conv2D), MaxPooling, Flatten, and Dense layers.

  #### Activation Functions: 
   ReLU for internal layers and Softmax for the final decision.

  #### Interface:
   Built with CustomTkinter, allowing users to draw a digit on a canvas and get real-time predictions.

### 2. Sentiment Analysis (NLP)
A text classification model designed to interpret emotional context from messages.
   #### Dataset: 
   Trained on a massive dataset of 1.6 million messages.
   #### Preprocessing:
   Includes cleaning, simplification (rooting), and TF-IDF vectorization
   #### Model:
   Logistic Regression optimized for high-speed classification.
   
## Real-World Validation (Facebook Data)
To verify the model's reliability, it was tested against actual Facebook comments to compare predictions with the "Ground Truth"
#### Total Valid Rows Analyzed: 921
#### Model Accuracy: 70.03%
#### Result: The AI's predictions closely match the real human sentiments, proving the model is ready for professional use cases.

## Requirements & Installation
To run this project, you need Python installed along with the following libraries:
### Core AI & Math
tensorflow
numpy
pandas
scikit-learn

### Computer Vision & GUI
opencv-python
customtkinter
Pillow

## Project Structure & File Descriptions
 ### Sentiment Analysis (NLP)
  #### interface_code.ipynb:
  This notebook provides a simple interface to test our trained Logistic Regression model on new, unseen text data.
  #### model_code.ipynb: 
  Contains the training process for the Sentiment Analysis model, including data cleaning and Logistic Regression training.
  #### visualisation.ipynb: 
  Dedicated to generating charts and comparing the AI's predictions with the Facebook Ground Truth data.
  #### trained_model.sav: 
  The saved state of the Sentiment Analysis model (Logistic Regression).
  #### vectorizer.sav:
  The TF-IDF vectorizer needed to transform new text into the numerical format the AI understands.
  #### facebook_sentiment_results: 
  The final output file containing the sentiments predicted by the AI for the Facebook comments.
  #### fb_sentiment: 
  The source file containing the original Facebook comments used for testing.
### Computer Vision (CNN)
 #### interface.ipynb :
 The main Python script for the CustomTkinter application. It handles the drawing canvas and real-time digit prediction.
 #### model.ipynb :
 The notebook containing the CNN architecture and training logic. It includes data normalization and the evaluation of the model's performance on the MNIST dataset.
 #### model/handwritten_model_v2.keras: 
 The pre-trained weights of the Convolutional Neural Network. Saving the model in .keras format allows the application to load the "brain" of the IA instantly without         needing to retrain it.
 
