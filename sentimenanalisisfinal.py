import requests
import json
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import csv
import os
import time
from firebase_admin import firestore
import firebase_admin
from firebase_admin import credentials

# Strore configuration credential for firebase and to connect to cloud database
cred = credentials.Certificate("./news-read-de276-firebase-adminsdk-voa3x-bebfcb3967.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Uncomment line below to download nltk requirements
nltk.download('punkt')

# Initiate variable to use into Machine Learning
factory = StemmerFactory()
stemmer = factory.create_stemmer()
Encoder = LabelEncoder()
Tfidf_vect = TfidfVectorizer()

# Configure data latih
DATA_LATIH = "./datasetgabung.csv"

def train_model():
    """Training step for Support Vector Classifier
    included data preprocessing using Sklearn module """
    print("train model")

    # Open Datasets
    datasets = pd.read_csv(DATA_LATIH, encoding='unicode_escape')
    print(datasets["label"].value_counts())

    # Text Normalization using PySastrawi(Word Stemmer for Bahasa Indonesia)
    lower = [stemmer.stem(row.lower()) for row in datasets["narasi"]]
    vectors = [word_tokenize(element) for element in lower]
    labels = datasets["label"]

    # Splitting Datasets for feeding to Machine Learning
    Train_X, Test_X, Train_Y, Test_Y = train_test_split(vectors, labels, test_size=0.25, stratify=labels)

    # Encoder for Data Label
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)

    # Create Tfidf Vector
    Tfidf_vect.fit(["".join(row) for row in lower])

    # Applying Tfidf for Training and Testing Features
    Train_X_Tfidf = Tfidf_vect.transform([" ".join(row) for row in Train_X])
    Test_X_Tfidf = Tfidf_vect.transform([" ".join(row) for row in Test_X])

    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = svm.SVC(C=1.0, kernel='linear', degree=1, gamma="auto", verbose=True)
    SVM.fit(Train_X_Tfidf, Train_Y)  # predict the labels on validation dataset

    # Use accuracy_score function to get the accuracy
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y) * 100)
    return SVM


def test_model(SVM):
    """Testing Machine Learning model using test
    datasets using the same method as training"""

    # Open data for testing the model
    print("test Model")
    datasets = pd.read_csv(DATA_UJI, encoding='unicode_escape')

    # Text Normalization using PySastrawi(Word Stemmer for Bahasa Indonesia)
    # Change all the text to lowercase
    # Text will be broke into set of words
    # Remove Stop word ("kamu yang aku benci" to "kamu aku benci")
    lower = [stemmer.stem(row.lower()) for row in datasets["narasi"]]
    vectors = [word_tokenize(element) for element in lower]

    # Applying Tfidf for Training and Testing Features
    # Word Ventorization
    Test_X_Tfidf = Tfidf_vect.transform([" ".join(row) for row in vectors])


    predictions_SVM = SVM.predict(Test_X_Tfidf)

    # Input to DataFrame
    data = {"ID": list(datasets["ID"]), "prediksi": predictions_SVM, "judul": list(datasets["Judul Berita"]),"narasi": list(datasets["narasi"])}
    hasil2 = pd.DataFrame(data, columns=["ID", "prediksi", "judul", "narasi"])
    hasil3 = hasil2.loc[hasil2["prediksi"] == 1]

    # Input dataframe to Firebase
    j = len(hasil3.index)
    for i in range(j):
        hasil4 = hasil3.iloc[i]
        hasil5 = hasil4.astype('string')
        convert_tojson = hasil5.to_json(orient="columns")
        parsed = json.loads(convert_tojson)
        db.collection(u'fakenews_db').document(u'fakenews_document'+str(i)).set(parsed)


    # hasil2.to_csv("./hasil uji model2.csv", index=False)


# Train Machine Learning model
SVM = train_model()

# Infite Loop
while True:

    # Scrapping www.kompas.com
    url = 'https://www.kompas.com/'
    html = requests.get(url)
    soup = BeautifulSoup(html.content, 'html.parser')
    populer = soup.find('div', {'class', 'most__wrap clearfix'})
    list_berita = populer.find_all('div', {'class', 'most__list clearfix'})

    # Get link that most popular news
    links = []
    for each in list_berita:
        links.append(each.a.get('href'))


    text = []
    dataset = []
    i = 1

    for link in links:
        page = requests.get(link + '?&page=all')
        soup = BeautifulSoup(page.content, 'html.parser')
        berita = soup.find('div', {'class', 'read__content'})
        nomor = i
        judul_berita = soup.find('h1', {'class', 'read__title'}).text
        unwanted_tag_p = soup.find_all('a', {'class', 'inner-link-baca-juga'})
        unwanted_tag_i = soup.find_all('i')

        # Drop unused tag p
        for x in unwanted_tag_p:
            x.extract()

        # Drop unused tag i
        for x in unwanted_tag_i:
            x.extract()

        # Drop unused text and merge string
        isi_berita = berita.text.replace('Baca juga:', '')
        isi_berita = berita.text.replace('KOMPAS.com -', '')
        isi_berita_tanpaenter = ' '.join(isi_berita.split())

        # Make temp table or data and will stored to dataframe
        dataset.append([nomor, judul_berita, isi_berita_tanpaenter])
        i = i + 1

    # Making Dataframe and store it to CSV file
    hasil = pd.DataFrame(dataset, columns=["ID", "Judul Berita", "narasi"])
    print("create scraping.csv")
    hasil.to_csv("./Scraping.csv", index=False)

    # Paused running code for 30 minuts
    # time.sleep(1800)

    # Configure data for data testing model to svm
    DATA_UJI = "./Scraping.csv"
    print("reading Data_uji")

    # Testing the data testing model with SVM
    test_model(SVM)
