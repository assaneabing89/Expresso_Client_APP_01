# Importing our dataset from csv file
import pandas as pd
from matplotlib import pyplot as plt
from ydata_profiling import ProfileReport
dataset = pd.read_csv('D:\Formation\Gomycode_datascience\Exo 26 Checkpoint\Expresso_churn_dataset.csv')
dataset = dataset.drop(['MRG'], axis = 1)
print(dataset.info())

# Missing Values
print(dataset.isnull().sum())

print(dataset.columns)

# Generate the profile report
profile = ProfileReport(dataset, title='Expresso_churn_dataset_Report')

# Display the report
# profile.to_notebook_iframe()
# Or generate an HTML report
# profile.to_file('D:\Formation\Gomycode_datascience\Exo 26 Checkpoint\Your_Report_Expresso_churn_dataset_V0.html')


# Missing Values
# Handling Missing Numerical Values

datasetNum = dataset.select_dtypes(exclude='object')
datasetNum = datasetNum.fillna(datasetNum.mean())
print(datasetNum.isnull().sum())

# Handling Missing Categorical Values
datasetCat = dataset.select_dtypes(include='object')
print(datasetCat.isnull().sum())

for column in datasetCat:
    datasetCat[column].fillna(datasetCat[column].mode()[0], inplace=True)
    datasetCat.isnull().sum()

print(datasetCat.isnull().sum())

# Encode categorical features
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
dataset_cat1 = encoder.fit_transform(datasetCat)
df2 = pd.DataFrame(dataset_cat1, columns=datasetCat.columns)
df2.head()

dataset_1 = pd.concat([df2, datasetNum], axis='columns')

print(dataset_1.head())
print(dataset_1.isnull().sum())

print(dataset_1['CHURN'].value_counts())


corr = dataset_1.corr( method = 'spearman')
print(corr)

# Generate the profile report
#profile = ProfileReport(dataset_1, title='Expresso_churn_dataset_Report')

# Display the report
#profile.to_notebook_iframe()
# Or generate an HTML report
#profile.to_file('D:\Formation\Gomycode_datascience\Exo 26 Checkpoint\Your_Report_Expresso_churn_dataset_V1.html')


import numpy as np; np.random.seed(0)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr ,annot=True,fmt=".1f")
plt.show()

df1 = dataset_1.drop([ 'user_id', 'REGION', 'TENURE', 'REGULARITY', 'TOP_PACK','FREQUENCE' ], axis = 1)
df1.shape

#Split your dataset to training and test sets

X = df1.drop(['CHURN'], axis = 1)
Y = df1['CHURN']
# machine learning handle arrays not data-frames
X = np.array(X).reshape(-1, 11)
Y = np.array(Y).reshape(-1, 1)
print(X.shape)
print(Y.shape)
X

#Répartion du Train_set et du Test_set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=30) #split our data with test size of 20%
print('Train_set:', x_train.shape)
print('Test_set:', x_test.shape)

#Choix de la méthode de Classification la mieux approprié

from sklearn.ensemble import RandomForestClassifier #Importing Random Forest Classifier

from sklearn import metrics
from sklearn.metrics import accuracy_score

# Create classifiers

model = RandomForestClassifier()
model.fit(x_train, y_train)  #Training our model
y_pred = model.predict(x_test)  #testing our model

acc = '{:.4f}'.format(accuracy_score(y_pred,y_test))
print(acc)

import joblib
joblib.dump( model,'D:\Formation\Gomycode_datascience\Exo 26 Checkpoint\Exo_26_Checkpoint_V2_model.joblib')



