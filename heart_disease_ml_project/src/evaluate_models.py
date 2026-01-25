
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score,matthews_corrcoef
import pandas as pd

def evaluate(models,X_test,y_test):
    rows=[]
    for n,m in models.items():
        y_pred=m.predict(X_test)
        y_prob=m.predict_proba(X_test)[:,1]
        rows.append({
            'Model':n,
            'Accuracy':accuracy_score(y_test,y_pred),
            'AUC':roc_auc_score(y_test,y_prob),
            'Precision':precision_score(y_test,y_pred),
            'Recall':recall_score(y_test,y_pred),
            'F1':f1_score(y_test,y_pred),
            'MCC':matthews_corrcoef(y_test,y_pred)
        })
    df=pd.DataFrame(rows)
    df.to_csv('results/evaluation_metrics.csv',index=False)
    return df
