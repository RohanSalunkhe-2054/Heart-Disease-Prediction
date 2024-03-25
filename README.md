# Heart Disease Prediction

This project aims to analyze a dataset containing various attributes related to heart health and build a machine learning model to predict the likelihood of heart disease based on those attributes.

## Dataset

The dataset used in this project is stored in a CSV file named `dataset.csv`. It contains the following columns:

- `age`: Age of the patient
- `sex`: Gender of the patient (0 = female, 1 = male)
- `cp`: Chest pain type (0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic)
- Add more description for other columns if needed...

## Visualization

The project includes visualizations of different distributions within the dataset, such as age distribution, sex distribution, chest pain type distribution, and target distribution (presence or absence of heart disease).

## Machine Learning Model

A logistic regression model is trained on the dataset to predict the presence or absence of heart disease. The dataset is split into training and testing sets, features are standardized, and then the model is trained and evaluated using accuracy score and classification report.

## Requirements

- Python 3.x
- pandas
- matplotlib
- scikit-learn
