import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Attendance': [60, 70, 75, 80, 85, 90, 95, 100],
    'Marks': [50, 55, 60, 65, 70, 75, 80, 85]
}

df = pd.DataFrame(data)
print(df)
print(df.head())

X = df[['Attendance']]
y = df['Marks']

from sklearn.model_selection import train_test_split
X_train , X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])


y_pred = model.predict(X_test)


#evaulation using the accuracy metrics
mse= mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("mse:", mse)
print("r2 score:", r2)
# Plotting the results

plt.scatter(X,y,color='green')
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.legend()
plt.grid(True)
plt.show()