import streamlit as st

from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

import seaborn as sns

st.set_page_config(layout='wide')

st.title("Random Forest")

X, y = make_classification(100, n_features=5, random_state=1234)

x_train, x_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=1235)

hyperparams = {}

# criterion
col1, col2 = st.columns(2)

hyperparams['n_estimators'] = st.slider('N estimators', 2, 5)

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
model = RandomForestClassifier(**hyperparams)

if st.button('Train'):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

try:
    fig, axes = plt.subplots(1, hyperparams['n_estimators'], figsize=(14,3))

    for idx, ax in enumerate(axes):
        plot_tree(model.estimators_[idx], ax=ax)
    st.pyplot(fig)
    col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
    with col1: 
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['recall'] = recall_score(y_test, y_pred)
        metrics['f1_score'] = f1_score(y_test, y_pred)
        metrics ['precision'] = precision_score(y_test, y_pred)

        metrics = pd.Series(metrics, name='value')
        st.dataframe(metrics)

    with col2:
        prec_recall_curve = PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
        st.pyplot(prec_recall_curve.figure_)

    with col3:
        auc_roc_curve = RocCurveDisplay.from_estimator(model, x_test, y_test)
        st.pyplot(auc_roc_curve.figure_)

    with col4:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, ax=ax, annot_kws={'fontsize':16}, cmap='viridis')
        ax.set_title("Confusion matrix", size=20)
        st.pyplot(fig)

except:
    
    st.warning("Entrena el modelo para mostrar las métricas")
