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
from datetime import date
import os
import time
from firebase_admin import firestore
import firebase_admin
from firebase_admin import credentials

cred = credentials.Certificate("./fakenewsapp-7c4fc-firebase-adminsdk-i2wov-d76adf2d90.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


today =date.today()
np.random.seed(500)

nltk.download('punkt')
factory = StemmerFactory()
stemmer = factory.create_stemmer()
Encoder = LabelEncoder()
Tfidf_vect = TfidfVectorizer()

#   Open file data latih
DATA_LATIH = "./datasetgabung.csv"

def train_model():
    """Training step for Support Vector Classifier
    included data preprocessing using Sklearn module """
    print("train model")
    # Open Datasets
    datasets = pd.read_csv(DATA_LATIH, encoding='unicode_escape')
    print(datasets["label"].value_counts())

    # Text Normalization using
    # PySastrawi(Word Stemmer for Bahasa Indonesia)
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
    print("test Model")
    datasets = pd.read_csv(DATA_UJI, encoding='unicode_escape')

    lower = [stemmer.stem(row.lower()) for row in datasets["narasi"]]
    vectors = [word_tokenize(element) for element in lower]

    Test_X_Tfidf = Tfidf_vect.transform([" ".join(row) for row in vectors])

    predictions_SVM = SVM.predict(Test_X_Tfidf)

    #Input to DataFrame
    data = {"ID": list(datasets["ID"]), "prediksi": predictions_SVM, "narasi": list(datasets["narasi"])}
    hasil = pd.DataFrame(data, columns=["ID", "prediksi", "narasi"])
    hasil_cleaning = hasil.loc[hasil["prediksi"] == 1]

    #Input to Firebase
    df_tostr = hasil_cleaning.astype("string")
    convert_tojson = df_tostr.to_json(orient= "columns")
    parsed = json.loads(convert_tojson)
    db.collection(u'fakenews_db').document(u'fakenews_document').set(parsed)

    #Drop Fake News then write to csv
    # hasil_cleaning.to_csv("./hasil cleaning.csv", index = False)

# Train Machine Learning model
SVM = train_model()

while True:
# Scrapping

    url             = 'https://www.kompas.com/'
    html            = requests.get(url)
    soup            = BeautifulSoup(html.content, 'html.parser')
    populer         = soup.find('div', {'class', 'most__wrap clearfix'})
    list_berita     = populer.find_all('div', {'class', 'most__list clearfix'})

    links   = []
    for each in list_berita:
        links.append(each.a.get('href'))

    text    = []
    i       = 1
    dataset = []
    for link in links:
        page    = requests.get(link + '?&page=all')
        soup    = BeautifulSoup(page.content, 'html.parser')
        berita  = soup.find('div', {'class', 'read__content'})
        nomor   = i
        judul_berita    = soup.find('h1', {'class', 'read__title'}).text
        unwanted_tag_p  = soup.find_all('a', {'class', 'inner-link-baca-juga'})
        unwanted_tag_i  = soup.find_all('i')
        for x in unwanted_tag_p:
            x.extract()
        for x in unwanted_tag_i:
            x.extract()
        isi_berita = berita.text.replace('Baca juga:', '')
        isi_berita_tanpaenter = ' '.join(isi_berita.split())
        dataset.append([nomor,judul_berita,isi_berita_tanpaenter])
        i = i + 1

    hasil = pd.DataFrame(dataset, columns=["ID", "Judul Berita", "narasi"])
    print("create scraping.csv")
    hasil.to_csv("./Scraping.csv", index=False)


    time.sleep(1800)
    DATA_UJI = "./Scraping.csv"
    print("reading Data_uji")
    
    # Testing the DATA_UJI with SVM
    test_model(SVM)
