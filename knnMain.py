import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer


class KNNClassifier:
    def __init__(self, k=3, p=2):
        self.k = k
        self.p = p

    # Fit, get_params, and set_params are required for GridSearchCV
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def get_params(self, deep=True):
        return {'k': self.k, 'p': self.p}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    # Main function to run KNN
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


# Setup the KNN model, split into training, cross validation with f1.
def trainKNN(X_train, y_train, k, p, split=True):
    knn = KNNClassifier(k=k, p=p)
    if split: # split if doing k-fold cross-validation
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Use f1 score for cross validation
    scorer = make_scorer(f1_score, average='weighted')
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring=scorer)

    # Train the model and get the precision, recall, and fscore
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_train)
    precision, recall, fscore, support = precision_recall_fscore_support(y_train, y_pred, average='weighted')
    return scores.mean(), precision, fscore 


"""
Main functions to train the KNN with grid search to find optimal values.
"""
def runKNN(filePath, targetColumn='target'):
    # Load the dataframe and the features and target
    df = pd.read_csv(filePath)
    df_train = pd.DataFrame(df)
    features = list(df_train.columns[:-1])
    X_train = df_train[features]
    y_train = df_train[targetColumn]

    # Create a scorer for the grid search. We will use the F1 score
    scorer = make_scorer(f1_score, average='weighted')
    param_grid = {
        'k': range(1, 20),
        'p': np.linspace(1, 2, 5)
    }

    # Perform a grid search to find the best k and p
    grid_search = GridSearchCV(KNNClassifier(), param_grid, scoring=scorer, cv=10)
    grid_search.fit(X_train, y_train)

    # Train the KNN model with the best k and p
    best_k = grid_search.best_params_['k']
    best_p = grid_search.best_params_['p']
    mean_score, percent_correct, fscore = trainKNN(X_train, y_train, best_k, best_p, split=False)
    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Mean f1-score with cross Validation: {mean_score}')
    print(f'Percent correct: {percent_correct}')
    print(f'f1-score: {fscore}')
