import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load your dataset
data = pd.read_excel('dataset.xlsx')

# Define features (X) and target (y)
X = data[['year', 'month', 'cp', 'yields', 'rainfall']]
y = data['price']

# Separate categorical and numerical columns
categorical_cols = ['cp']
numerical_cols = ['year', 'month', 'yields', 'rainfall']

# Preprocessing: One-Hot Encoding for categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Create a pipeline that includes preprocessing and model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', LinearRegression())])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the trained model to a .pkl file
model_filename = 'crop_price_prediction_model.pkl'
joblib.dump(model, model_filename)
