# customer_churn_prediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         customer_churn_prediction and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── customer_churn_prediction   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes customer_churn_prediction a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```
Thank you for pointing that out! Here's the updated `README.md` with the missing section included:

```markdown
# Customer Churn Prediction

This project is focused on predicting customer churn using machine learning models. It involves data preprocessing, exploratory data analysis (EDA), model training, evaluation, and making predictions using various classification algorithms.

## 1. Importing the Dependencies

First, we import all the necessary libraries required for the project:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
```

### Key Dependencies:
- **NumPy and Pandas** for data manipulation.
- **Matplotlib and Seaborn** for data visualization.
- **Scikit-learn** for machine learning algorithms and metrics.
- **Imbalanced-learn (SMOTE)** for handling class imbalance.
- **Pickle** for saving and loading models.

## 2. Data Loading and Understanding

The dataset is loaded using Pandas, and an initial exploration is conducted to understand the structure and content of the data.

```python
# load the csv data to a pandas dataframe
df = pd.read_csv("/content/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.shape
df.head()
pd.set_option("display.max_columns", None)
df.head(2)
df.info()

# dropping customerID column as this is not required for modelling
df = df.drop(columns=["customerID"])
df.head(2)

# inspecting unique values in categorical columns
print(df["gender"].unique())
print(df["SeniorCitizen"].unique())

# printing the unique values in all the columns
numerical_features_list = ["tenure", "MonthlyCharges", "TotalCharges"]
for col in df.columns:
    if col not in numerical_features_list:
        print(col, df[col].unique())
        print("-"*50)

print(df.isnull().sum())
```

### Insights:
1. The `customerID` column is dropped since it is not relevant for modeling.
2. There are no missing values in the dataset except for the `TotalCharges` column.
3. Missing values in the `TotalCharges` column are replaced with 0.0.
4. Class imbalance is identified in the target variable `Churn`.

## 3. Exploratory Data Analysis (EDA)

In this section, we perform a thorough analysis of the numerical and categorical features to understand their distributions and relationships.

### Numerical Features - Analysis

Histograms are plotted to show the distribution of numerical features like `tenure`, `MonthlyCharges`, and `TotalCharges`.

```python
def plot_histogram(df, column_name):
    plt.figure(figsize=(5, 3))
    sns.histplot(df[column_name], kde=True)
    plt.title(f"Distribution of {column_name}")

    # calculate the mean and median values for the columns
    col_mean = df[column_name].mean()
    col_median = df[column_name].median()

    # add vertical lines for mean and median
    plt.axvline(col_mean, color="red", linestyle="--", label="Mean")
    plt.axvline(col_median, color="green", linestyle="-", label="Median")
    plt.legend()
    plt.show()

plot_histogram(df, "tenure")
plot_histogram(df, "MonthlyCharges")
plot_histogram(df, "TotalCharges")
```

### Box Plot for Numerical Features

We use box plots to examine the distribution of numerical features and identify potential outliers.

```python
def plot_boxplot(df, column_name):
    plt.figure(figsize=(5, 3))
    sns.boxplot(y=df[column_name])
    plt.title(f"Box Plot of {column_name}")
    plt.ylabel(column_name)
    plt.show()

plot_boxplot(df, "tenure")
plot_boxplot(df, "MonthlyCharges")
plot_boxplot(df, "TotalCharges")
```

### Correlation Heatmap

We plot a heatmap to visualize the correlations between numerical features.

```python
plt.figure(figsize=(8, 4))
sns.heatmap(df[["tenure", "MonthlyCharges", "TotalCharges"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
```

### Categorical Features - Analysis

We use count plots to visualize the distribution of categorical features.

```python
object_cols = df.select_dtypes(include="object").columns.to_list()
object_cols = ["SeniorCitizen"] + object_cols

for col in object_cols:
    plt.figure(figsize=(5, 3))
    sns.countplot(x=df[col])
    plt.title(f"Count Plot of {col}")
    plt.show()
```

## 4. Data Preprocessing

We preprocess the data by performing label encoding on the target column and other categorical features.

### Label Encoding the Target Column

The `Churn` column is encoded as 0 and 1.

```python
df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})
```

### Label Encoding of Categorical Features

We apply label encoding to the categorical columns.

```python
# Apply LabelEncoder to all categorical columns
label_encoder = LabelEncoder()

categorical_columns = df.select_dtypes(include="object").columns
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])
```

## 5. Handling Class Imbalance

SMOTE (Synthetic Minority Over-sampling Technique) is used to handle class imbalance in the target variable.

```python
smote = SMOTE()
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_res, y_res = smote.fit_resample(X, y)
```

## 6. Model Training and Evaluation

We train multiple machine learning models and evaluate them based on accuracy, confusion matrix, and classification report.

### Model Selection:
- **DecisionTreeClassifier**
- **RandomForestClassifier**
- **XGBClassifier**

### Model Training and Evaluation:

```python
# Train the models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier()
}

for model_name, model in models.items():
    model.fit(X_res, y_res)
    y_pred = model.predict(X_test)
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
```

## 7. Model Prediction (New Data)

To make predictions with a new input, we use the saved encoder and trained model to process the input data and output the prediction.

```pythoninput_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}


input_data_df = pd.DataFrame([input_data])

with open("encoder.pkl", "rb") as f:
  encoders = pickle.load(f)


# encode categorical featires using teh saved encoders
for column, encoder in encoders.items():
  input_data_df[column] = encoder.transform(input_data_df[column])

# make a prediction
prediction = loaded_model.predict(input_data_df)
pred_prob = loaded_model.predict_proba(input_data_df)

print(prediction)

# results
print(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
print(f"Prediciton Probability: {pred_prob}")
```

## 8. Saving the Model

The trained model is saved using `pickle` for future predictions.

```python
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
```

## Conclusion

- The dataset is preprocessed by handling missing values and encoding categorical variables.
- Several machine learning models are trained, and their performance is evaluated.
- The final model is saved for future predictions.

## Future Work

- Experiment with hyperparameter tuning to improve model performance.
- Deploy the model as a web application for real-time predictions.

---

Feel free to modify or extend the project based on your requirements.
```

This updated `README.md` file now includes the new section for making predictions with new input data, which utilizes both the saved encoders and the trained model to make churn predictions.
--------

