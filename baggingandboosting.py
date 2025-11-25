from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
     

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
     

bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)

bagging_model.fit(X_train, y_train)
y_pred_bag = bagging_model.predict(X_test)

boosting_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

boosting_model.fit(X_train, y_train)
y_pred_boost = boosting_model.predict(X_test)

print("========== BAGGING RESULTS ==========")
print("Accuracy:", accuracy_score(y_test, y_pred_bag))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bag))
print("Classification Report:\n", classification_report(y_test, y_pred_bag))

print("\n========== BOOSTING RESULTS ==========")
print("Accuracy:", accuracy_score(y_test, y_pred_boost))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_boost))
print("Classification Report:\n", classification_report(y_test, y_pred_boost))
     