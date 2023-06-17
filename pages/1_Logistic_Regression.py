import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import make_classification


st.set_page_config(layout='wide')

X, y = make_classification(100, n_features=5)

x_train, x_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=1235)
# teoría



#Hiperparámetros

#Datos

#Resultados

#Métricas

#Curvas