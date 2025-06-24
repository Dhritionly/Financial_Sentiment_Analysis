
# Financial Sentiment Analysis using News Headlines

## Project Overview

This project focuses on analyzing sentiment from daily news headlines to predict stock market movements. By leveraging Natural Language Processing (NLP) techniques, specifically the Bag-of-Words model and a RandomForestClassifier, this solution aims to classify news sentiment and correlate it with market trends.

## Features

-   **Data Ingestion:** Reads daily news headlines and associated stock market labels (e.g., UP/DOWN).
    
-   **Text Preprocessing:** Cleans and transforms raw news headlines, including punctuation removal and lowercasing.
    
-   **Feature Engineering:** Utilizes `CountVectorizer` with n-grams to convert text data into numerical features (Bag-of-Words).
    
-   **Sentiment Classification:** Employs a `RandomForestClassifier` to predict market sentiment (represented by the 'Label' column in the dataset) based on news headlines.
    
-   **Performance Evaluation:** Reports on model accuracy using a confusion matrix, accuracy score, and classification report.
    

## Technologies Used

-   **Python**
    
-   **Pandas:** For data manipulation and analysis.
    
-   **Scikit-learn:** For machine learning algorithms, including:
    
    -   `CountVectorizer` for text feature extraction.
        
    -   `RandomForestClassifier` for sentiment classification.
        
    -   `classification_report`, `confusion_matrix`, `accuracy_score` for model evaluation.
        

## Dataset

The project uses a CSV file named `Data.csv`, which contains daily news headlines (Top1 to Top25) and a corresponding 'Label' column indicating market movement (likely 0 for down/neutral and 1 for up). The dataset is split into training and testing sets based on date.

## Project Structure

-   `Stock Sentiment Analysis.ipynb`: The main Jupyter Notebook containing all the code for data loading, preprocessing, model training, and evaluation.
    
-   `Data.csv`: (Assumed) The dataset used for the project.
    

## Setup and Installation

To run this project, you need Python and the following libraries installed:

1.  **Clone the repository (or download the files):**
    
    ```
    git clone https://github.com/your-username/stock-sentiment-analysis.git
    cd stock-sentiment-analysis
    
    ```
    
    (Note: Replace `your-username` with your actual GitHub username if you create a repo for this.)
    
2.  **Install dependencies:**
    
    ```
    pip install pandas scikit-learn
    
    ```
    

## Usage

1.  **Ensure `Data.csv` is in the same directory** as `Stock Sentiment Analysis.ipynb`.
    
2.  **Open the Jupyter Notebook:**
    
    ```
    jupyter notebook "Stock Sentiment Analysis.ipynb"
    
    ```
    
3.  **Run all cells** in the notebook. The output will display:
    
    -   The head of the processed dataset.
        
    -   The confusion matrix.
        
    -   The accuracy score.
        
    -   The classification report (precision, recall, f1-score) for the sentiment predictions.
        

## Results

The current model achieves an accuracy of approximately **84.13%** on the test dataset, as indicated by the classification report.

```
[[139  47]
 [ 13 179]]
0.8412698412698413
              precision    recall  f1-score   support

           0       0.91      0.75      0.82       186
           1       0.79      0.93      0.86       192

   micro avg       0.84      0.84      0.84       378
   macro avg       0.85      0.84      0.84       378
weighted avg       0.85      0.84      0.84       378

```

## Future Enhancements

-   Explore more advanced NLP techniques (e.g., TF-IDF, Word Embeddings, or Transformer-based models from Hugging Face for more nuanced sentiment capture).
    
-   Integrate with real-time news APIs for live sentiment analysis.
    
-   Combine sentiment features with other technical indicators for enhanced prediction models.
    
-   Implement a time-series forecasting component to predict actual stock price movements based on sentiment.
    
-   Develop a user interface or dashboard for visualizing sentiment trends and predictions.
    
-   Experiment with deep learning models (e.g., LSTMs, GRUs) for sequential data.
    

## Contact

For any questions or collaborations, feel free to reach out.

Dhritionly

GitHub Profile (Please replace with your actual GitHub profile link)
