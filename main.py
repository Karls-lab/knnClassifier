import knnMain
import os
import sys

diabetesPath = os.path.join(os.path.dirname(__file__), 'data', 'diabetes_cleaned.csv')
clevelandPath = os.path.join(os.path.dirname(__file__), 'data', 'cleveland_cleaned.csv')

# check if cleaned data exists 
if not os.path.exists(clevelandPath):
    print("Please run heartDiseasePreprocess.ipynb")
    sys.exit()    

if not os.path.exists(diabetesPath):
    print("Please run diabetesPreprocess.ipynb")
    sys.exit()

print("\nLocated Datasets! Running KNN...")

# Now run the knn model for all datasets
print("\nRunning KNN for heart disease")
knnMain.runKNN(clevelandPath, targetColumn='num')

print("\nRunning KNN for diabetes")
# knnMain.runKNN(diabetesPath, targetColumn='Outcome')

