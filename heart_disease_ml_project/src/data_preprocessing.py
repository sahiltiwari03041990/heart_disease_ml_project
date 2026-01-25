import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# def load_and_preprocess(path):
#     df=pd.read_csv(path)
#     X=df.drop('target',axis=1)
#     y=df['target']
#     scaler=StandardScaler()
#     X_scaled=scaler.fit_transform(X)
#     return train_test_split(X_scaled,y,test_size=0.2,random_state=42,stratify=y)


def load_and_preprocess(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    print (df.shape)
    # Check if any class has only 1 member
    if df['target'].value_counts().min() < 2:
        # Remove the class with only 1 member
        min_class = df['target'].value_counts().idxmin()
        df = df[df['target'] != min_class]
    
    # Split the data into training and testing sets
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_and_preprocess(
    "/Volumes/nonprod_u2prepay/prepay_common/playground/Tsahil/heart.csv"
)



