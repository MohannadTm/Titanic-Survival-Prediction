import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/titanic.csv')

df = df.drop(columns=['Cabin'])
df['Age'] = df['Age'].fillna(df['Age'].mean())
df = df.dropna(subset=['Embarked'])

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

df['FamilySize'] = df['SibSp'] + df['Parch']

X = df[['Pclass','Sex','Age','Fare','Embarked','FamilySize']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))
