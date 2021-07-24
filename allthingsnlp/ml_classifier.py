import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

import xgboost as xgb

def compare_models(new):
    
    X = np.array(new['cleaned_hm'])
    y = np.array(new[['predicted_category']])
    
    # Creating model dictionary
    #model_list=['Logistic Regression','Decision Tree']
    model_list=['Logistic Regression','Decision Tree','Random Forest','CatBoost']
    model_results = dict()
    for model in model_list:
        model_results[model]={'Accuracy':list(),'Precision':list(),'Recall':list(),'F1':list()}
        
    skf = StratifiedKFold(n_splits=2,shuffle=True)
    split=0  
    for train_index, test_index in skf.split(X, y):
        split+=1
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        vectorizer = TfidfVectorizer()
        train_features = vectorizer.fit_transform(X_train)
        test_features = vectorizer.transform(X_test)
        
        for model in model_list:
            print(f'Training {model} for split {split}')
            
            if model == 'Logistic Regression': 
                lr_clr = linear_model.LogisticRegression()
                lr_clr.fit(train_features, y_train)
                y_pred=lr_clr.predict(test_features)
                
            elif model == 'Decision Tree':
                dt_clf = DecisionTreeClassifier(random_state=0)
                dt_clf.fit(train_features, y_train)
                y_pred=dt_clf.predict(test_features)
                
            elif model == 'Random Forest':
                rf_clf = RandomForestClassifier(random_state=42)
                rf_clf.fit(train_features, y_train)
                y_pred=rf_clf.predict(test_features)         
                
            elif model == 'XGBoost':
                xgb_clf = xgb.XGBClassifier(random_state=42)
                xgb_clf.fit(train_features, y_train)
                y_pred=xgb_clf.predict(test_features)
                
            elif model == 'CatBoost':
                cat_clf=CatBoostClassifier(random_state=42)
                cat_clf.fit(train_features, y_train)
                y_pred=cat_clf.predict(test_features)   
            
            
            model_results[model]['Accuracy'].append(accuracy_score(y_test, y_pred))
            model_results[model]['Precision'].append(precision_recall_fscore_support(y_test, y_pred, average='macro')[0])
            model_results[model]['Recall'].append(precision_recall_fscore_support(y_test, y_pred, average='macro')[1])
            model_results[model]['F1'].append(precision_recall_fscore_support(y_test, y_pred, average='macro')[2])
    
        
    output_df=pd.DataFrame(columns=['Model','Accuracy','Precision','Recall','F1'])
    for model in model_results.keys():
        df_length = len(output_df)
        Accuracy=round(sum(model_results[model]['Accuracy'])/len(model_results[model]['Accuracy']),2)
        Precision=round(sum(model_results[model]['Precision'])/len(model_results[model]['Precision']),2)
        Recall=round(sum(model_results[model]['Recall'])/len(model_results[model]['Recall']),2)
        F1=round(sum(model_results[model]['F1'])/len(model_results[model]['F1']),2)
        output_df.loc[df_length] = [model,Accuracy,Precision,Recall,F1]
        
    return(output_df)
        