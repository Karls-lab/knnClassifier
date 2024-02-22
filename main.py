import knnMain
import os

"""
Perform KNN grid search on specified dataset
"""

heartDiseasePath = os.path.join(os.path.dirname(__file__), 'heartDisease', 'cleveland_cleaned.csv')
knnMain.runKNN(heartDiseasePath, targetColumn='num')

diabetesPath = os.path.join(os.path.dirname(__file__), 'diabetes', 'diabetes_preprocessed.csv')
knnMain.runKNN(diabetesPath, targetColumn='Outcome')
