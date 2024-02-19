import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

class KNNClassifier:
    def __init__(self, k=3, p=2):
        self.k = k
        self.p = p

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for i, x in X_test.iterrows():
            # Calculate Minkowski Distance between x and all training points
            distances = np.sum(np.abs(self.X_train - x) ** self.p, axis=1) ** (1 / self.p)
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train.iloc[nearest_indices]
            unique_labels, counts = np.unique(nearest_labels, return_counts=True)
            most_common_label = unique_labels[np.argmax(counts)]
            predictions.append(most_common_label)


        return predictions

df = pd.read_csv('cleveland_cleaned.csv')
df_train = pd.DataFrame(df)
features = list(df_train.columns[:-1])
X_train = df_train[features]
y_train = df_train['num']
# print(X_train)
# print(y_train)

def trainKNN(X_train, y_train, k, p):
    # take 80% of the data as training data and 20% as testing data
    X_train, X_test, y_train, y_test = train_test_split(df_train.drop('num', axis=1), df_train['num'], test_size=0.2, random_state=42)

    knn = KNNClassifier(k=k, p=p)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    f1 = f1_score(y_test, predictions)
    return f1

scores = {}
for k in range(1, 20):
    for p in np.linspace(1, 2, 10):
        scores[(k, p)] = trainKNN(X_train, y_train, k, p)

top_kp = sorted(scores, key=scores.get, reverse=True)[:5]
for kp in top_kp:
    print(f'k={kp[0]}, p={kp[1]}: f1={scores[kp]:.4f}')