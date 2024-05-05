import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Sample data (replace with your actual data)
# Assume you have a DataFrame named 'data' containing features related to courses and user preferences
# and a column 'rating' indicating user ratings for each course
data = pd.read_csv("data.csv")

# Separate features and target variable
X = data.drop(columns=['rating'])
y = data['rating']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor as a recommender model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Initialize SHAP explainer
explainer = shap.Explainer(model, X_train)

# Generate SHAP values for test data
shap_values = explainer.shap_values(X_test)

# Visualize SHAP summary plot
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)