import numpy as np 
import pandas as pd 
import matplotlib as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix

# Sample data
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Pass':          [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        }
df = pd.DataFrame(data)

x = df[['Hours_Studied']]
y = df['Pass']   # data splitted by dataframe
x_train , x_test, y_train , y_test = train_test_split(x,x,test_size=0.3 , random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# Evaluation
accuracy = accuracy_score(y_test, y_pred)
confusion_matrix_result = confusion_matrix(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

print("CONFUSION_M:", confusion_matrix)
print("Accuracy:", accuracy)
print("C_REPORT:", classification_report_result)


# Plotting the results.....
plt.scatter(x,y,color='blue')
plt.plot(x,y_pred,color='red',linewidth=2)




