import pandas as pd
from sklearn.metrics import confusion_matrix

df = pd.read_excel('iris.xls')

x = df.iloc[:,1:4].values
y = df.iloc[:,4:].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)


# 1. Logistic Regression

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

con_matrix = confusion_matrix(y_test, y_pred)
print('LogisticRegression')
print(con_matrix)


# 2. KNN

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

con_matrix = confusion_matrix(y_test, y_pred)
print('KNeighborsClassifier')
print(con_matrix)


# 3. SVC (SVM Classifier)

from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

con_matrix = confusion_matrix(y_pred, y_test)
print('SVM')
print(con_matrix)


# 4. Naive Bayes

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

con_matrix = confusion_matrix(y_test, y_pred)
print('Naive Bayes')
print(con_matrix)


# 5. Decision Tree

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

con_matrix = confusion_matrix(y_test, y_pred)
print('DTC')
print(con_matrix)


# 6. Random Forest

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

con_matrix = confusion_matrix(y_test, y_pred)
print('RFC')
print(con_matrix)


# 7. ROC, TPR, FPR degerleri

y_proba = rfc.predict_proba(X_test)
print(y_test)
print(y_proba[:,0])

from sklearn import metrics
fbr, tpr, thold = metrics.roc_curve(y_test, y_proba[:,0])
print(fbr)
print(tpr)









