# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Check input files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load datasets
data_train = pd.read_csv('data/customer_churn_dataset-training-master.csv')
data_test  = pd.read_csv('data/customer_churn_dataset-testing-master.csv')

# Initial exploration
data_train.columns
data_test.columns
data_train.shape
data_test.shape
data_train.info()
data_test.info()
data_train.isnull().sum()
data_train[data_train['CustomerID'].isnull()]
data_train.dropna(inplace=True)
data_train.duplicated().sum()
data_test.duplicated().sum()
data_train.describe()
data_train.describe(include='O')

# Plot churn distribution
plt.figure(figsize=(12,6))
sns.countplot(x=data_train['Churn'])
plt.title('Frequance Churn',fontsize=20,fontweight='bold')
plt.ylabel('Frequance',fontsize=16)
plt.xlabel('Churn',fontsize=16)
plt.show()

plt.figure(figsize=(12,6))
sns.countplot(x=data_test['Churn'])
plt.title('Frequance Churn',fontsize=20,fontweight='bold')
plt.ylabel('Frequance',fontsize=16)
plt.xlabel('Churn',fontsize=16)
plt.show()

# Handle class imbalance
class0 = data_train[data_train['Churn']==0]
class1 = data_train[data_train['Churn']==1]
print(f"size of Class(0): {len(class0)} | size of Class(1): {len(class1)}")

samples_needed = len(class1) - len(class0)
numeric_cols = ['Age','Tenure','Usage Frequency','Support Calls','Payment Delay','Total Spend','Last Interaction']
samples = class0.sample(n=samples_needed,replace=True,random_state=42)
augmented = samples.copy()

for col in numeric_cols:
    augmented[col] += np.random.normal(0,data_train[col].std(),size=samples_needed)

df = pd.concat([data_train,augmented],ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

plt.figure(figsize=(12,6))
sns.countplot(x=df['Churn'])
plt.title('Class distribution after augmentation:',fontsize=20,fontweight='bold')
plt.ylabel('Frequance',fontsize=16)
plt.xlabel('Churn',fontsize=16)
plt.show()

# Encoding categorical features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['Gender'] = le.fit_transform(df['Gender'])
df['Subscription Type'] = le.fit_transform(df['Subscription Type'])
df['Contract Length'] = le.fit_transform(df['Contract Length'])

data_test['Gender'] = le.fit_transform(data_test['Gender'])
data_test['Subscription Type'] = le.fit_transform(data_test['Subscription Type'])
data_test['Contract Length'] = le.fit_transform(data_test['Contract Length'])

# Distribution plots
n_cols = 3
n_rows = (len(df.columns[1:12]) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
axes = axes.flatten()

for i, col in enumerate(df.columns[1:12]):
    sns.histplot(df[col], ax=axes[i], kde=True, bins=30, color='mediumorchid')
    mean_val = df[col].mean()
    median_val = df[col].median()
    axes[i].axvline(mean_val, color='blue', linestyle='--', label=f'Mean: {mean_val:.1f}')
    axes[i].axvline(median_val, color='red', linestyle='-', label=f'Median: {median_val:.1f}')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].legend()

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Scaling
from sklearn.preprocessing import StandardScaler, RobustScaler
cols = ['Usage Frequency','Age','Tenure']
sd = StandardScaler()
df[cols] = sd.fit_transform(df[cols])
data_test[cols] = sd.fit_transform(data_test[cols])

cols = ['Payment Delay','Total Spend','Last Interaction']
rb = RobustScaler()
df[cols] = rb.fit_transform(df[cols])
data_test[cols] = rb.fit_transform(data_test[cols])

# Correlation heatmap
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),annot=True)
plt.show()

# Splitting features and labels
X_train = df.drop(['CustomerID','Churn','Contract Length'],axis=1)
X_test = data_test.drop(['CustomerID','Churn','Contract Length'],axis=1)
y_train = df['Churn']
y_test = data_test['Churn']

print(f"{'\U0001F539 Training Features:':<25} {X_train.shape}")
print(f"{'\U0001F539 Testing Features:':<25} {X_test.shape}")
print(f"{'\U0001F538 Training Labels:':<25} {y_train.shape}")
print(f"{'\U0001F538 Testing Labels:':<25} {y_test.shape}")

# Model 1: Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print(model.score(X_train,y_train))
print(model.score(X_test,y_test))

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("Accuracy : ",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

con = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(12,6))
sns.heatmap(con,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix')
plt.show()

# Model 2: Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
dt.fit(X_train,y_train)

print(dt.score(X_train,y_train))
print(dt.score(X_test,y_test))

y_pred = dt.predict(X_test)
print('Accuracy : ',accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

con = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(12,6))
sns.heatmap(con,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix')
plt.show()

# Model 3: Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)
rf.fit(X_train,y_train)

print(rf.score(X_train,y_train))
print(rf.score(X_test,y_test))

y_pred = rf.predict(X_test)
print('Accuracy : ',accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

con = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(12,6))
sns.heatmap(con,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix')
plt.show()
