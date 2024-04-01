import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv('C:\\Users\\ASUS\\Downloads\\dataset.csv')

# Plotting age distribution
plt.figure(figsize=(10, 6))
plt.hist(data['age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plotting sex distribution
plt.figure(figsize=(6, 6))
sex_counts = data['sex'].value_counts()
plt.pie(sex_counts, labels=['Male', 'Female'], autopct='%1.1f%%', colors=['lightcoral', 'lightskyblue'])
plt.title('Sex Distribution')
plt.show()

# Plotting chest pain type distribution
plt.figure(figsize=(8, 6))
cp_counts = data['cp'].value_counts()
cp_labels = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
plt.bar(cp_labels, cp_counts, color='lightgreen')
plt.title('Chest Pain Type Distribution')
plt.xlabel('Chest Pain Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Plotting target distribution
plt.figure(figsize=(6, 6))
target_counts = data['target'].value_counts()
plt.pie(target_counts, labels=['No Heart Disease', 'Heart Disease'], autopct='%1.1f%%', colors=['lightcoral', 'lightskyblue'])
plt.title('Target Distribution')
plt.show()

# Separate features and target variable
features = data.drop(columns=['target'])  # All columns except 'target' are features
target = data['target']  # 'target' column is the target variable

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training - Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, y_train)

# Model evaluation - Logistic Regression
y_pred_logistic = logistic_model.predict(X_test_scaled)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print("Logistic Regression Model Accuracy:", accuracy_logistic)
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_logistic))

# Model training - Decision Tree Classifier
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train_scaled, y_train)

# Model evaluation - Decision Tree Classifier
y_pred_decision_tree = decision_tree_model.predict(X_test_scaled)
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
print("Decision Tree Classifier Model Accuracy:", accuracy_decision_tree)
print("Decision Tree Classifier Classification Report:\n", classification_report(y_test, y_pred_decision_tree))
