import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , r2_score ,mean_squared_error

df = pd.read_csv("Housing.csv")

#EDA + pre proccessing

df['mainroad'] = df['mainroad'].map({'yes': 1 , 'no': 0})
df['guestroom'] = df['guestroom'].map({'yes': 1 , 'no': 0})
df['basement'] = df["basement"].map({'yes': 1 , 'no': 0})
df['hotwaterheating'] = df['hotwaterheating'].map({'yes': 1 , 'no': 0})
df['airconditioning'] = df['airconditioning'].map({'yes': 1 , 'no': 0})
df['prefarea'] = df['prefarea'].map({'yes': 1 , 'no': 0})
df = pd.get_dummies(df, columns = ['furnishingstatus'])
df.drop( columns = ['furnishingstatus_unfurnished', 'furnishingstatus_semi-furnished'],inplace=True) #low correlation with price

#print(df.describe())


#Checking correlatiob between diffeerent features
df.plot(kind = 'scatter', y = 'price' , x = 'area')
sns.heatmap(df.corr(), annot= True)
#plt.show()

#Train test split

X = df.drop('price', axis=1)
y = df['price'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Metrics:")
print("R² Score:", r2_score(y_test, y_pred)) #accounts for n% variance in data
print("MAE:", mean_absolute_error(y_test,y_pred)) #How far off the prediction estimates are in given units
print("MSE:", mean_squared_error(y_test,y_pred)) #Square error less forgiving metric
