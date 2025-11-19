import pandas as pd
import numpy as np

df=pd.read_csv('BankNote_Authentication.csv')

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

X.head()
y.head(20)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)
print(score)
import pickle
pickle_out=open('classifier.pkl','wb')
pickle.dump(classifier, pickle_out)
pickle_out.close()
import numpy as np
classifier.predict([[2,3,4,1]])