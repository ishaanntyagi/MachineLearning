import pandas as pd
import sklearn
s=pd.Series([10,20],index=[";",":"])
print(s)

data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Marks": [85, 90, 95]
}

df = pd.DataFrame(data)
print(df)
print(df.shape)  
print(df.dtypes)



dataC=pd.read_csv(r"C:\Users\ishaa\Downloads\sample_students.csv")
print(dataC.head())
print(dataC.describe())
print(dataC.info())
print(dataC.isnull().sum())


# label encoding as the coloumns like gender must be encoded 
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
updates=df['Gender'] = le.fit_transform(df['Gender'])
print(df,updates)