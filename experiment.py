import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer


# selecting data for model
train_data = '/home/user/Desktop/titanic/train.csv'
test_data = '/home/user/Desktop/titanic/test.csv'
train = pd.read_csv(train_data)
test = pd.read_csv(test_data)

df = train.append(test, ignore_index=True)

# print(df.head())
def process_data(df):
    # specify features
    df_features = ['Pclass', 'Fare', 'Cabin', 'Name', 'SibSp', 'Parch', 'Age', 'Sex', 'Embarked']

    X = df[df_features]

    Pclass_matrix = pd.get_dummies(X['Pclass'])
    X = pd.concat([X, Pclass_matrix], axis=1)
    X.drop('Pclass', axis=1, inplace=True)

    fare_list = X["Fare"].dropna()
    average_fare = fare_list.mean()
    X['Fare'] = X['Fare'].fillna(average_fare)

    # sns.barplot(x="Embarked", y="Survived", data=df)
    # plt.show()
    X['Embarked'] = X['Embarked'].fillna('N')
    embarked_matrix = pd.get_dummies(X['Embarked'])
    X = pd.merge(X, embarked_matrix, left_index=True, right_index=True)
    del X['Embarked']

    age_list = X["Age"].dropna()
    average_age = age_list.mean()
    X['Age'] = X['Age'].fillna(average_age)

    X.loc[X["Sex"] == "male", "Sex"] = 0
    X.loc[X["Sex"] == "female", "Sex"] = 1

    X["Cabin"] = X["Cabin"].fillna('Z')
    X["Cabin"] = X["Cabin"].apply(lambda x: x[0])
    # sns.barplot(x="Cabin", y="Fare", data=X)
    # plt.show()
    cabin_matrix = pd.get_dummies(X["Cabin"])
    X = pd.merge(X, cabin_matrix, left_index=True, right_index=True)
    del X["Cabin"]

    X["Name"] = X["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
    X['Name'] = X['Name'].replace(['Lady', 'the Countess', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    X['Name'] = X['Name'].replace('Mlle', 'Miss')
    X['Name'] = X['Name'].replace('Ms', 'Miss')
    X['Name'] = X['Name'].replace('Mme', 'Mrs')
    lb = LabelBinarizer()
    name_matrix = pd.DataFrame(lb.fit_transform(X["Name"]))
    name_matrix.columns = lb.classes_
    X = pd.merge(X, name_matrix, left_index=True, right_index=True)
    del X["Name"]

    return X

processed_df = process_data(df)

X = processed_df[:len(train)]
y = train["Survived"]
X_pred = processed_df[len(train):]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# building model
titanic_model = RandomForestClassifier(random_state=1)

titanic_model.fit(X_train, y_train)

y_pred = titanic_model.predict(X_test)

print(accuracy_score(y_test, y_pred)*100.0)

ids = test['PassengerId']

predictions = titanic_model.predict(X_pred)
output = pd.DataFrame({'PassengerId': ids, 'Survived': predictions})
output.to_csv('prediction.csv', index=False)

