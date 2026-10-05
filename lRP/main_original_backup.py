from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Load the dataset
ad_data = pd.read_csv(Path(__file__).with_name("advertising.csv"))


# Inspect the dataset
print(ad_data.head())
print(ad_data.info())
print(ad_data.describe())
print(ad_data.isnull().sum())


# Age histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=ad_data, x="Age", bins=30, kde=True)
plt.title("Distribution of Age")
plt.show()


# Area Income versus Age
sns.jointplot(data=ad_data, x="Age", y="Area Income")
plt.show()


# Daily Time Spent on Site versus Age
sns.jointplot(
    data=ad_data,
    x="Age",
    y="Daily Time Spent on Site",
    kind="kde",
    fill=True,
)
plt.show()


# Daily Time Spent on Site versus Daily Internet Usage
sns.jointplot(
    data=ad_data,
    x="Daily Time Spent on Site",
    y="Daily Internet Usage",
)
plt.show()


# Pairplot
plot_columns = [
    "Daily Time Spent on Site",
    "Age",
    "Area Income",
    "Daily Internet Usage",
    "Male",
    "Clicked on Ad",
]

sns.pairplot(
    ad_data[plot_columns],
    hue="Clicked on Ad",
    corner=True,
)
plt.show()


# Select features and target
feature_columns = [
    "Daily Time Spent on Site",
    "Age",
    "Area Income",
    "Daily Internet Usage",
    "Male",
]

X = ad_data[feature_columns]
y = ad_data["Clicked on Ad"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)


# Create and train the model
model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "logistic_regression",
            LogisticRegression(max_iter=1000, random_state=42),
        ),
    ]
)

model.fit(X_train, y_train)


# Make predictions
predictions = model.predict(X_test)


# Evaluate
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
