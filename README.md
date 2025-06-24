# Stock Market Sentiment Analysis using News Headlines

## Project Overview

This project focuses on analyzing sentiment from daily news headlines to predict stock market movements. By leveraging Natural Language Processing (NLP) techniques, specifically the Bag-of-Words model and a RandomForestClassifier, this solution aims to classify news sentiment and correlate it with market trends.

**A significant portion of the Python code for this project was generated with the assistance of Generative AI**, enabling efficient development and implementation of machine learning concepts.

## Features

-   **Data Ingestion:** Reads daily news headlines and associated stock market labels (e.g., UP/DOWN).
    
-   **Text Preprocessing:** Cleans and transforms raw news headlines, including punctuation removal and lowercasing.
    
-   **Feature Engineering:** Utilizes `CountVectorizer` with n-grams to convert text data into numerical features (Bag-of-Words).
    
-   **Sentiment Classification:** Employs a `RandomForestClassifier` to predict market sentiment (represented by the 'Label' column in the dataset) based on news headlines.
    
-   **Performance Evaluation:** Reports on model accuracy using a confusion matrix, accuracy score, and classification report.
    

## Technologies Used

-   **Python**
    
-   **Pandas:** For data manipulation and analysis.
    
-   **Scikit-learn:** For machine learning algorithms (`CountVectorizer`, `RandomForestClassifier`, `classification_report`, `confusion_matrix`, `accuracy_score`).
    
-   **Generative AI (e.g., OpenAI models):** Utilized as a powerful coding assistant for generating Python scripts and implementing NLP/ML components.
    

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

## How Generative AI Assisted in Coding

Generative AI played a crucial role in developing this project by providing code snippets and logic for various NLP and machine learning tasks. This significantly accelerated the development process, especially for complex components where specialized knowledge might otherwise be required.

Here's how Generative AI was used to obtain the code for this project:

1.  **Clear Problem Definition:** The process begins by clearly defining the specific task or problem at hand. For example, "How do I load a CSV into a pandas DataFrame and inspect its first few rows?"
    
2.  **Specific Prompting:** Instead of vague questions, precise prompts were given to the AI. The more detail provided (e.g., desired libraries, data structure, expected output), the better the generated code.
    
    -   **For Data Loading:**
        
        -   _Prompt Example:_ "Write Python code using pandas to load a CSV file named 'Data.csv' with 'ISO-8859-1' encoding. Then, display the first 5 rows of the DataFrame."
            
        -   _Result:_ This would generate code similar to: `df=pd.read_csv('Data.csv', encoding = "ISO-8859-1")` and `df.head()`.
            
    -   **For Text Preprocessing (Removing Punctuation and Lowercasing):**
        
        -   _Prompt Example:_ "How can I remove all non-alphabetic characters from text columns (0 to 24) in a pandas DataFrame called `data` and convert them to lowercase? Also, rename the columns to '0', '1', '2', etc., for easy access."
            
        -   _Result:_ This would yield code similar to:
            
            ```
            data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
            list1= [i for i in range(25)]
            new_Index=[str(i) for i in list1]
            data.columns= new_Index
            for index in new_Index:
                data[index]=data[index].str.lower()
            
            ```
            
    -   **For Bag-of-Words Feature Extraction:**
        
        -   _Prompt Example:_ "Generate Python code using `sklearn.feature_extraction.text.CountVectorizer` to implement a Bag-of-Words model with 2-word n-grams. Fit it on a list of headlines called `headlines`."
            
        -   _Result:_ This would produce:
            
            ```
            from sklearn.feature_extraction.text import CountVectorizer
            countvector=CountVectorizer(ngram_range=(2,2))
            traindataset=countvector.fit_transform(headlines)
            
            ```
            
    -   **For RandomForestClassifier and Prediction:**
        
        -   _Prompt Example:_ "Provide Python code using `sklearn.ensemble.RandomForestClassifier` to train a classification model with 200 estimators and 'entropy' criterion. Then, use this trained model to make predictions on a new dataset called `test_dataset`."
            
        -   _Result:_ This would give:
            
            ```
            from sklearn.ensemble import RandomForestClassifier
            randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
            randomclassifier.fit(traindataset,train['Label'])
            predictions = randomclassifier.predict(test_dataset)
            
            ```
            
    -   **For Model Evaluation Metrics:**
        
        -   _Prompt Example:_ "Show me how to calculate and print the confusion matrix, accuracy score, and classification report in Python using `sklearn.metrics` for `test['Label']` and `predictions`."
            
        -   _Result:_ This would generate:
            
            ```
            from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
            matrix=confusion_matrix(test['Label'],predictions)
            print(matrix)
            score=accuracy_score(test['Label'],predictions)
            print(score)
            report=classification_report(test['Label'],predictions)
            print(report)
            
            ```
            
3.  **Iterative Refinement and Debugging:** The generated code was then integrated into the Jupyter Notebook. If errors occurred or the output wasn't as expected, the AI was prompted with the error messages or refined requirements until the desired functionality was achieved. This iterative process of prompting, executing, and refining was key to building the project components.
    

## Future Enhancements

-   Explore more advanced NLP techniques (e.g., TF-IDF, Word Embeddings, or Transformer-based models from Hugging Face for more nuanced sentiment capture).
    
-   Integrate with real-time news APIs for live sentiment analysis.
    
-   Combine sentiment features with other technical indicators for enhanced prediction models.
    
-   Develop a user interface or dashboard for visualizing sentiment trends and predictions.
    
-   Experiment with deep learning models (e.g., LSTMs, GRUs) for sequential data.
    

## Contact

For any questions or collaborations, feel free to reach out.

Dhritionly

[GitHub Profile](https://github.com/Dhritionly/)
