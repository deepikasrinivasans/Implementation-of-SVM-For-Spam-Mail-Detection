# EX-09 Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1.Start

step 2.Import the required packages.

step 3.Import the dataset to operate on.

step 4.Split the dataset.

step 5.Predict the required output.

step 6.Stop.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: DEEPIKA S
RegisterNumber:  212222230028
*/

import pandas as pd
data = pd.read_csv("spam.csv",encoding="Windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### data.head():
![1MM](https://github.com/deepikasrinivasans/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393935/c0918e79-253b-4757-bcec-11a6d83df3f6)
### data.info():
![2MM](https://github.com/deepikasrinivasans/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393935/1728c2d9-e57c-4129-a08d-bbd4925471b6)
### data.isnull()sum():
![3MM](https://github.com/deepikasrinivasans/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393935/a85370eb-9913-4fea-96ee-3531eea2256b)
### y_predict:
![4MM](https://github.com/deepikasrinivasans/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393935/ae42ce1c-691c-4024-bc35-c77f0327ec66)
### Accuracy:
![5MM](https://github.com/deepikasrinivasans/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393935/82afd1b4-a1bc-4fe2-88ef-f436e5906dfb)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
