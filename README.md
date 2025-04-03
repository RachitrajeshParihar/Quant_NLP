# Quant_NLP
MLP Mixer for Financial Sentiment Analysis:

Project Overview:
This project involves financial sentiment analysis using tweets. The goal is to process financial text data, extract meaningful features using BERT embeddings, and classify sentiment using an MLP-Mixer model. The pipeline includes data scraping, preprocessing, model training, and evaluation to achieve high accuracy in sentiment classification.

Project Structure:
-Data Scraping: Extracts financial tweets and sentiment labels.

-Preprocessing: Converts text into numerical representations using BERT embeddings.

-Model Training: Implements the MLP-Mixer architecture for sentiment classification.

-Evaluation: Uses multiple metrics such as accuracy, RMSE, and correlation.

Technologies Used:
*Python, PyTorch, Transformers (Hugging Face), NumPy, Pandas
*BeautifulSoup and Requests for Web Scraping
*BERT for Embedding Generation
*MLP-Mixer Architecture for Classification

Step-by-Step Implementation

1. Data Scraping
Financial tweets were scraped using BeautifulSoup and Requests. A dataset was extracted from "TimKoornstra/financial-tweets-sentiment", and URLs were parsed to fetch complete tweet content. The data was cleaned using regular expressions, removing links and special characters.

Code Used:
requests for fetching web pages
BeautifulSoup for HTML parsing
re module for text cleaning
multiprocessing for parallelized scraping
Final Output: sentiment001.csv

2. Data Preprocessing
To prepare the scraped dataset for training, tweets were converted into embeddings using BERT (bert-base-uncased). The embeddings were then reshaped into 16×16 patches to feed into the MLP-Mixer model.

Steps:

Load dataset and remove NaN values
Tokenize text using BERT tokenizer
Extract embeddings from the last hidden state of the BERT model
Reshape embeddings into patches for input into MLP-Mixer
Final Output: X.npy and y.npy (processed dataset)

3. Training MLP-Mixer
An MLP-Mixer model was trained on the processed data. It consists of token mixing and channel mixing layers to learn feature representations.

Model Architecture:
Token Mixing: Captures interactions between patches
Channel Mixing: Learns spatial relationships within patches
Fully Connected Layer: Outputs sentiment classification

Training Details:
Dataset Split: 80% Training, 20% Validation
Optimizer: Adam (lr=1e-3)
Loss Function: CrossEntropyLoss (for classification)
Batch Size: 32
Early Stopping: Stops training when validation loss does not improve for 3 epochs
Saved Model: mlp_mixer_best.pth

4. Model Evaluation
The trained model was evaluated using multiple metrics:
Accuracy (for classification tasks)
Root Mean Square Error (RMSE)
Pearson Correlation Coefficient (to measure the relationship between predictions and actual values)
Results Printed in Console:
Accuracy measures the percentage of correctly classified tweets
RMSE evaluates model error magnitude
Correlation measures how well predictions align with true sentiment

How to Run the Project:
*Clone the repository
*Install dependencies:
  pip install torch transformers pandas numpy requests beautifulsoup4

*Run the scraping script:
  python scrape.py 

*Run the preprocessing script:
  python preprocess.py  

*Train the model:
  python train.py  
  
Author:
This project was implemented as part of an effort to explore MLP-Mixer models for NLP-based sentiment classification in the financial domain.
-by RACHIT PARIHAR              email- rachit.parihar0418@gmail.com
    SHREYAS SHASHI KUMAR GOWDA  email- mynameisshreyasshashi@gmail.com


