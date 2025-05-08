import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle
import streamlit as st


df=pd.read_csv(r"C:\Users\Plhv\Downloads\iris\iris.csv")


# EDA
# Pop up Baisc Information
print("Dataset Info")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nSummary Statistics:")
print(df.describe())


# Preprocess the data
X=df.drop('species',axis=1)
y=df['species']


## Convert species to numerical labels (optional, for model compatibility)
# Setosa: 0, Versicolor: 1, Virginica: 2
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for deployment
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Step 4: Train and evaluate multiple models
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC()
}

# Dictionary to store model performance
performance = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    performance[name] = accuracy
    print(f"\n{name}:")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {name}')
    plt.savefig(f'cm_{name.lower().replace(" ", "_")}.png')  # Save for documentation
    plt.show()

# Identify the best model
best_model_name = max(performance, key=performance.get)
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name} with accuracy {performance[best_model_name]:.2f}")

# Save the best model
pickle.dump(best_model, open('iris_model.pkl', 'wb'))
pickle.dump(le, open('label_encoder.pkl', 'wb'))  # Save label encoder for deployment

# Step 5: Hyperparameter Tuning for KNN (Optional)
param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

print(f"\nKNN Hyperparameter Tuning Results:")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.2f}")
