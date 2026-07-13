# Import required libraries for data analysis and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the wine dataset into a Pandas DataFrame
df = pd.read_csv("dataset/wine_clean.csv")

# Display the first five rows of the dataset
print(df.head())

# Check dataset information such as data types and missing values
print(df.info())

# Generate statistical summary of numerical columns
print(df.describe())

# Check for missing values in each column
print(df.isnull().sum())

# Count the number of samples in each wine class
print(df["Class"].value_counts())

# Visualize the distribution of wine classes
plt.figure(figsize=(6,4))
sns.countplot(x="Class", data=df)
plt.title("Wine Class Distribution")
plt.xlabel("Wine Class")
plt.ylabel("Count")
plt.show()

# Plot correlation heatmap
plt.figure(figsize=(12,8))

sns.heatmap(df.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Plot histograms for all features
df.hist(figsize=(16,12))
plt.tight_layout()
plt.show()

# Separate input features and target variable
X = df.drop("Class", axis=1)
y = df["Class"]

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

# Train Logistic Regression model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))

from sklearn.tree import DecisionTreeClassifier
# Train Decision Tree model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))

# Train Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))


from sklearn.svm import SVC

# Train Support Vector Machine model
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))

import joblib
# Save the trained Random Forest model
joblib.dump(rf, "model/wine_model.pkl")

print("Model saved successfully!")