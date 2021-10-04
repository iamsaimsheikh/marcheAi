from django.shortcuts import render
from django.http import HttpResponse

from rest_framework.views import APIView
from django.http.response import JsonResponse
from rest_framework.parsers import JSONParser
from rest_framework.response import Response

import json
import pandas as pd
recommendations = pd.read_csv('recommendation.csv')
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import joblib


X_train, X_test, y_train, y_test = train_test_split(recommendations['Keywords'], recommendations['Recommendations'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

# Create your views here.


class tags(APIView):
    def get(self, request):
        return Response('halos')

    def post(self, request):
        tags = JSONParser().parse(request)
        data = tags['data']
        length = len(data)
        response = []
        for i in range(length):
            response.append(clf.predict(count_vect.transform([data[i]])))
            print(response[i])

        npDec = np.array([response])
        result = npDec.flatten().tolist()

        return Response(result)
