import streamlit as st
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')

X, y = make_classification(100, n_features=5)

x_train, x_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=1235)
# teoría


#Hiperparámetros
hyperparams = {}

# criterion
col1, col2 = st.columns(2)

with col1: 
    hyperparams['criterion'] = st.selectbox("Criterion", ['gini', 'entropy', 'log_loss'])

    #max_depth
    hyperparams['max_depth'] = st.slider("Max depth", 2, 10)

# min_samples_split

with col2: 
    hyperparams['min_samples_split'] = st.slider('Min samples split', 2, 10)

    #min_samples_leaf
    hyperparams['min_samples_leaf'] = st.slider('Min samples leaf', 2, 10)

#Entrenar
model = DecisionTreeClassifier(**hyperparams)

if st.button('Train'):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

#Árbol

try:
    col3, col4 = st.columns(2)
    with col3:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            plot_tree(model)
            st.pyplot(fig)
            #Métricas
    with col4:
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['recall'] = recall_score(y_test, y_pred)
        metrics['f1_score'] = f1_score(y_test, y_pred)
        metrics ['precision'] = precision_score(y_test, y_pred)

        metrics = pd.Series(metrics, name='value')
        st.dataframe(metrics)

except :
    st.warning("Entrena el modelo para mostrar las métricas")


