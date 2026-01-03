import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("d137464c-e35c-48ec-8124-9a81866280c0.csv")

# Rename columns if needed (adjust names to match dataset)
# Example assumption:
# sqft, bedrooms, bathrooms, year_built, price

X = data[['sqft_living', 'bedrooms', 'bathrooms', 'yr_built']]
y = data['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open("house_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
