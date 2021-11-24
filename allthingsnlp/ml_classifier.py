import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

import xgboost as xgb


class ml_classifier:
    
    
    """
    This module trains and tunes different machine learning models and compares the results.
    """
    
    def __init__(self,input_df,text_feature,target):
        self.input_df=input_df
        self.text_feature=text_feature
        self.target=target
        
        
    # Function to train and compare different models
    def compare_models(self,optimize='F1'):
    
        X = np.array(self.input_df[self.text_feature])
        y = np.array(self.input_df[self.target])
        
        # Creating model dictionary
        model_list=['Logistic Regression','Decision Tree']
        #model_list=['Random Forest','CatBoost']
        
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
        
        # Sort based on optimization parameter
        output_df=output_df.sort_values(by=[optimize],ascending=False)  
        return(output_df)
            
    def train_model(self,estimator='lr'):
            
        """
        Train and hypertune estimators
        
        """

        train_features = np.array(self.input_df[self.text_feature])
        
        vectorizer = TfidfVectorizer()
        train_features = vectorizer.fit_transform(train_features)
        y = np.array(self.input_df[self.target])
        
        # define evaluation
        cv = StratifiedKFold(n_splits=2, random_state=1,shuffle=True)
        
        if estimator=='lr':
            model = linear_model.LogisticRegression()
            
            # define search space
            space = dict()
            space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
            space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
            space['C'] = loguniform(1e-5, 100)
                
        elif estimator=='dt':
            model = DecisionTreeClassifier()
            space = {'criterion':['gini','entropy'],'max_depth':range(1,10),'min_samples_leaf':range(1,5),'min_samples_split':range(1,10)}
            
        elif estimator=='rf':
            model= RandomForestClassifier()
            space={'bootstrap': [True, False],'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                   'max_features': ['auto', 'sqrt'],
                   'min_samples_leaf': [1, 2, 4],
                   'min_samples_split': [2, 5, 10],
                   'n_estimators': [200, 400]}
        
        elif estimator=='xgb':
            model= xgb.XGBClassifier()
            space= {'min_child_weight': [1, 5, 10],'gamma': [0.5, 1, 1.5, 2, 5],'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'max_depth': [3, 4, 5],'n_estimators': [200, 400]}
            
        search = RandomizedSearchCV(model, space, n_iter=10, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)
        # execute search
        result = search.fit(train_features, y)
            
        return(result)

                
                
        
        

        

