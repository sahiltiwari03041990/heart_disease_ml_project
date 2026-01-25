
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib, os

def train_all_models(X_train,y_train):
    models={
        'Logistic Regression':LogisticRegression(max_iter=1000),
        'Decision Tree':DecisionTreeClassifier(),
        'KNN':KNeighborsClassifier(),
        'Naive Bayes':GaussianNB(),
        'Random Forest':RandomForestClassifier(n_estimators=100),
        'XGBoost':XGBClassifier(eval_metric='logloss',use_label_encoder=False)
    }
    os.makedirs('models',exist_ok=True)
    for n,m in models.items():
        m.fit(X_train,y_train)
        joblib.dump(m,f"models/{n.replace(' ','_').lower()}.pkl")
    return models
