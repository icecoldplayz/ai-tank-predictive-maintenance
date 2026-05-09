import numpy as np
import pandas as pd

np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "age": np.random.randint(1, 20, n),
    "material": np.random.choice([0, 1, 2], n),  # 0=steel, 1=aluminum, 2=plastic
    "temperature": np.random.uniform(10, 50, n),
    "humidity": np.random.uniform(20, 100, n),
    "chemical_exposure": np.random.uniform(0, 10, n),
    "usage_frequency": np.random.randint(1, 50, n),
    "last_cleaned_days": np.random.randint(1, 365, n)
})

# Create corrosion formula (this is your "fake reality")
data["corrosion_level"] = (
    data["age"] * 3 +
    data["humidity"] * 0.4 +
    data["chemical_exposure"] * 4 +
    data["last_cleaned_days"] * 0.08 +
    data["temperature"] * 0.5 +
    data["usage_frequency"] * 0.2
)

# Add some randomness
data["corrosion_level"] += np.random.normal(0, 5, n)

# Clamp between 0 and 100
data["corrosion_level"] = np.clip(data["corrosion_level"], 0, 100)

# Create classification target
data["needs_cleaning"] = (data["corrosion_level"] > 60).astype(int)

print(data.head())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = data.drop(["corrosion_level", "needs_cleaning"], axis=1)
y = data["needs_cleaning"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Features (inputs)
X = data.drop(["corrosion_level", "needs_cleaning"], axis=1)

# Target (what we predict)
y = data["corrosion_level"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
reg_model = RandomForestRegressor()
reg_model.fit(X_train, y_train)

# Predictions
preds = reg_model.predict(X_test)

# Evaluate
error = mean_absolute_error(y_test, preds)

print("Mean Absolute Error:", error)