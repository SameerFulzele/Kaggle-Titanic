import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import category_encoders as ce
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn import svm

import warnings; warnings.simplefilter('ignore')
import statsmodels.api as sm 
import gc
import re


train =  pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


##############################################################################################################################################
#FEATURE ENGINEERING 

def Feature_Engineering(df):
     
    df.Embarked = df.Embarked.fillna('S') 
    # As only 2 values are missing I am assuming Port of Embarkation is Southampton(S) as majority of people aboarded from S
    
    
    df.Cabin = df.Cabin.fillna('0')
    df['Has_Cabin'] = df["Cabin"].apply(lambda x: 0 if x == '0' else 1)
    
    
    df['Family_Size'] = df.SibSp + df.Parch + 1
    
    
    df['Alone'] = df["Family_Size"].apply(lambda x: 1 if x == 1 else 0)
    
    
    df['Age'][np.isnan(df['Age'])] = np.random.randint(df.Age.mean() - df.Age.std() , df.Age.mean() + df.Age.std(), size = df.Age.isnull().sum())

 
    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
   
    
    encoder=ce.OneHotEncoder(cols='Embarked',handle_unknown='return_nan',return_df=True,use_cat_names=True)
    con= encoder.fit_transform(df['Embarked'])
    df= pd.concat([df,con.iloc[:,1:]], axis=1)
    #df['Embarked'] = df['Embarked'].map( {'S': 1, 'C': 2, 'Q': 3} ).astype(int)
    df= df.drop('Embarked',axis=1)

    
    df['Age_Bins'] = 0
    df.loc[ df['Age'] <= 16, 'Age_Bins'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age_Bins'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age_Bins'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age_Bins'] = 3
    df.loc[ df['Age'] > 64, 'Age_Bins'] = 4 
    
    
    df['Fare_Bins'] = 0
    df.loc[ df['Fare'] <= 10, 'Fare_Bins'] = 0   
    df.loc[(df['Fare'] > 10) & (df['Fare'] <= 40), 'Fare_Bins'] = 1  
    df.loc[(df['Fare'] > 40) & (df['Fare'] <= 80), 'Fare_Bins'] = 2  
    df.loc[ df['Fare'] > 80, 'Fare_Bins'] = 3 
    
    
    return df    

 ##############################################################################################################################################


#FEATURE SELECTION


df_train = train.copy()
df_train = Feature_Engineering(df_train)
df_train = df_train.drop(['Name','Ticket','Cabin','Age','SibSp','Parch','Fare','PassengerId'],axis=1)


df_test = test.copy()
df_test = Feature_Engineering(df_test)
df_test = df_test.drop(['Name','Ticket','Cabin','Age','SibSp','Parch','Fare','PassengerId'],axis=1)


X = df_train.drop( 'Survived',axis = 1)
Y = df_train['Survived']


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state= 100)


##############################################################################################################################################
#MODEL SELECTIONS AND INITIALIZING PARAMETERS TO TEST

# Wr are going to use nested dictionary having key = names, values = Model class and parameters as per sklearn

algos = {


		'Logistic Regression' : {'model': LogisticRegression,
                            'parameters': {'penalty': ['l2','none'],
                                           'fit_intercept' : [True,False]
                                              }
                          		},


	'Naive Bayes multinomial' : {'model': MultinomialNB,
                              'parameters': {'alpha': [0.01,0.1,0.5,1,2,5,10,20],
                                            'fit_prior': ['random', 'cyclic']
                                         }
                          		},
    
        
	'Decision Tree Classifier': {'model': tree.DecisionTreeClassifier, 
                            'parameters': {'criterion' : ['gini','entropy'],
                                           'splitter': ['best','random']
                                          }
                          		},
         
       	'SVM': {'model':svm.SVC, 
                           'parameters': {'kernel' :['linear']
                                         }
                         		}
        	
        }

############################################################################################################################################
#MODELLING IN OOP 
seed = 100

class Model():

    seed = 100 
    def __init__(self,dataframe,X,Y,algo_dictionary, problem_type):
        self.df = dataframe
        self.algos = algo_dictionary
        self.problem_type = problem_type
        self.X = X
        self.Y = Y
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X,self.Y,test_size=0.2,random_state= self.seed)


    def Gridsearch(self,name,update_best_features = None):


        for key,value in self.algos[name].items():

            cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=self.seed)
            gs =  GridSearchCV(value['model'](), value['parameters'], cv=cv, return_train_score=False)
            gs.fit(self.X,self.Y)

            if update_best_features is True:
                value['parameters'] = gs.best_params_


        print({'model': key,'best_score': gs.best_score_,'best_params': gs.best_params_})


    def Gridsearch_all(self,update_best_features = None):


        table = []
        for key,value in self.algos.items():

            cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state= self.seed)
            gs =  GridSearchCV(value['model'](), value['parameters'], cv=cv, return_train_score=False)
            gs.fit(self.X,self.Y)
            table.append({'model': key,'best_score': gs.best_score_,'best_params': gs.best_params_})

            if update_best_features is True:
                value['parameters'] = gs.best_params_


        table = pd.DataFrame(table)
        print(table)


    def train(self,algorithm_name):


        parameters = self.algos[algorithm_name]['parameters']
        self.algos[algorithm_name]['model'] = self.algos[algorithm_name]['model'](**parameters)
        self.self.algos[algorithm_name]['model'].fit(self.X_train,self.Y_train)


        # for key,value in algos[name].items():
        # 	parameters = value['parameters']
        # 	value['model'] = value['model'](**parameters)
        # 	value['model'].fit(x_train, y_train)


        predictions = self.algos[algorithm_name]['model'].predict(self.X_test)
        if self.problem_type == 'Classification':
            print('Classification report after predicting on testing data for', algorithm_name)
            print(classification_report(self.Y_test, predictions))
            print('TN,FN')
            print('FP,TP')
            print(confusion_matrix(self.Y_test, predictions))


    def train_all(self):
        for key,value in self.algos.items():
            parameters = value['parameters']
            value['model'] = value['model'](**parameters)
            value['model'].fit(self.X_train, self.Y_train)

            predictions = value['model'].predict(self.X_test)
            if self.problem_type == 'Classification':
                print('Classification report after predicting on testing data for', key)
                print(classification_report(self.Y_test, predictions))
                print('TN,FN')
                print('FP,TP')
                print(confusion_matrix(self.Y_test, predictions))
                print('\n')

            if self.problem_type == 'Regression':
                pass


    def predict(self,x_input,algorithm_name):
        predictions = self.algos[algorithm_name]['model'].predict(x_input)
        return predictions


    def feature_importances(self,x,y):
        pass


    def permutation_importances(self,x,y):
        pass


    def remove_algo(self, name):
        del self.algos[str(name)]


    def add_algo(self,dict):
        self.algos.update(dict)


    def reset_seed(self , new_seed):
        self.seed = new_seed


    def report(self):
        if self.problem_type == 'Regression':
            pass

        elif self.problem_type == 'Classification':
            pass

        else:
            print('Please define problem_type')


mod = Model(df_train,X,Y,algos,'Classification')

mod.Gridsearch_all(update_best_features= True)

mod.train_all()


