# Fake News Detection System

A machine learning pipeline to classify news articles as fake or real using text data. This project combines data preprocessing, feature extraction, and multiple classification models to optimize the detection of misinformation.

## 📂 Dataset

The dataset used in this project is the **Fake and Real News Dataset by Clément V.**, which includes:

- `Fake.csv` — Fake news articles
- `True.csv` — Real news articles

You can download it from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

---

## 🛠 Features

- Combines news titles and content for analysis  
- Cleans text by removing punctuation, stopwords, and normalizing case  
- Uses **TF-IDF** for feature extraction  
- Implements and compares multiple classifiers:
  - Logistic Regression
  - Decision Tree
  - Random Forest  
- Evaluates models using accuracy, classification report, and confusion matrix  
- Saves the trained model and vectorizer for future inference  
- Provides an example of predicting new news text  

---

## 🚀 Technologies Used

- Python
- scikit-learn
- Pandas
- NumPy
- NLTK (Natural Language Toolkit)
- Jupyter Notebook / Google Colab
- joblib (for model saving/loading)

---

## 📦 How to Run

1. Download the dataset from Kaggle:  
   [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

2. Upload `Fake.csv` and `True.csv` to your Colab environment.

3. Run the provided notebook script to:
   - Preprocess the data
   - Train the models
   - Evaluate performance
   - Save the final model and vectorizer
   - Test with new inputs

---

## 📈 Results

- Achieved high accuracy with Random Forest  
- Demonstrated effective feature extraction using TF-IDF  
- Enabled easy inference with saved models

---

## 📌 Next Steps

- Experiment with additional text preprocessing techniques (stemming, lemmatization)  
- Tune hyperparameters and explore other classifiers  
- Build a web interface using Flask or Streamlit  
- Deploy the model as an API

