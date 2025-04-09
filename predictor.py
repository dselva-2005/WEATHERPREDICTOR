import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Load the dataset
df = pd.read_csv("seattle-weather.csv")

# 2. Clean and preprocess
df = df.dropna()
df['date'] = pd.to_datetime(df['date'])

# 3. Feature engineering
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
df['temp_avg'] = (df['temp_max'] + df['temp_min'])/2

# 4. Define features and target
X = df[['precipitation', 'temp_avg', 'wind', 'month', 'day', 'dayofweek']]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['weather'])
joblib.dump(label_encoder, 'label_encoder.pkl')
# 5. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 7. Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test Mean Squared Error: {mse:.2f}")

# 8. Save the model
joblib.dump(model, "weather_model.pkl")
print("Model saved to weather_model.pkl")
