import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ==============================
# 1. LOAD DATASET
# ==============================
df = pd.read_csv("data/amazon.csv")

 # <-- your uploaded file

print("Before Cleaning:")
print(df.head())
print(df.shape)

# ==============================
# 2. DATA CLEANING
# ==============================

# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna("Unknown")
    else:
        df[col] = df[col].fillna(df[col].median())

print("\nAfter Cleaning:")
print(df.shape)

# Save cleaned data
df.to_csv("cleaned_amazon_data.csv", index=False)

# ==============================
# CONVERT PRICE, RATING COLUMNS TO NUMERIC
# ==============================

df['discounted_price'] = df['discounted_price'].str.replace('₹','').str.replace(',','')
df['actual_price'] = df['actual_price'].str.replace('₹','').str.replace(',','')
df['discount_percentage'] = df['discount_percentage'].str.replace('%','')
df['rating_count'] = df['rating_count'].str.replace(',','')

df['discounted_price'] = pd.to_numeric(df['discounted_price'], errors='coerce')
df['actual_price'] = pd.to_numeric(df['actual_price'], errors='coerce')
df['discount_percentage'] = pd.to_numeric(df['discount_percentage'], errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')

df = df.fillna(0)



# ==============================
# 3. DATA INFO & SUMMARY
# ==============================
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ==============================
# 4. DATA VISUALIZATION
# (Only if Sales column exists)
# ==============================
if 'Sales' in df.columns:

    plt.figure()
    plt.hist(df['Sales'], bins=10)
    plt.xlabel("Sales")
    plt.ylabel("Frequency")
    plt.title("Sales Distribution")
    plt.show()

# ==============================
# 5. FEATURE SELECTION
# ==============================
numeric_df = df.select_dtypes(include=['number'])

if 'Sales' in numeric_df.columns and numeric_df.shape[1] > 1:
    corr = numeric_df.corr()
    print("\nCorrelation with Sales:")
    print(corr['Sales'])

# ==============================
# 6. DATA NORMALIZATION
# ==============================
if numeric_df.shape[1] > 1:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)

    print("\nNormalized Data (first 5 rows):")
    print(scaled_df.head())

# ==============================
# 7. MACHINE LEARNING (Linear Regression)
# ==============================
# ==============================
# MACHINE LEARNING (Linear Regression)
# Predict rating
# ==============================

df_ml = df[['discounted_price', 'actual_price', 
            'discount_percentage', 'rating_count', 'rating']]

X = df_ml.drop('rating', axis=1)   # Features
y = df_ml['rating']               # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)

print("Linear Regression model trained successfully")




# ==============================
# MODEL TESTING
# ==============================

# Predict using test data
y_pred = model.predict(X_test)

print("\nPredicted Ratings (first 5):")
print(y_pred[:5])

print("\nActual Ratings (first 5):")
print(y_test.values[:5])

# ==============================
# RESULT ANALYSIS
# ==============================

import pandas as pd
import matplotlib.pyplot as plt

# Create comparison table
results = pd.DataFrame({
    'Actual Rating': y_test.values,
    'Predicted Rating': y_pred
})

print("\nActual vs Predicted Ratings (first 10):")
print(results.head(10))

# Plot comparison
plt.figure()
plt.plot(results['Actual Rating'].values[:20], label='Actual')
plt.plot(results['Predicted Rating'].values[:20], label='Predicted')
plt.xlabel("Sample Index")
plt.ylabel("Rating")
plt.title("Actual vs Predicted Ratings")
plt.legend()
plt.show()



# Make predictions on test data
y_pred = model.predict(X_test)

# ==============================
# MODEL EVALUATION
# ==============================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)


# ==============================
# PERFORMANCE IMPROVEMENT
# ==============================

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

# ----- Feature Selection -----
selector = SelectKBest(score_func=f_regression, k=3)
X_new = selector.fit_transform(X, y)

selected_columns = X.columns[selector.get_support()]
print("Selected Best Features:", selected_columns)

# Split again using selected features
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X[selected_columns], y, test_size=0.2, random_state=42
)

# ----- Hyperparameter Tuning (Linear Regression) -----
params = {
    'fit_intercept': [True, False]
}

grid = GridSearchCV(LinearRegression(), params, cv=5)
grid.fit(X_train_new, y_train_new)

print("Best Parameters:", grid.best_params_)

# Train improved model
best_model = grid.best_estimator_
y_pred_new = best_model.predict(X_test_new)

# Evaluate Improved Model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae_new = mean_absolute_error(y_test_new, y_pred_new)
rmse_new = np.sqrt(mean_squared_error(y_test_new, y_pred_new))
r2_new = r2_score(y_test_new, y_pred_new)

print("\nImproved Model Performance:")
print("MAE:", mae_new)
print("RMSE:", rmse_new)
print("R2 Score:", r2_new)
 

import joblib

# Save the trained model to file
joblib.dump(model, "amazon_rating_prediction_model.pkl")

print("Model saved successfully!")

# -------- To load and use later --------

# Load saved model
loaded_model = joblib.load("amazon_rating_prediction_model.pkl")

# Example prediction using loaded model
sample_prediction = loaded_model.predict(X_test[:5])
print("Prediction using loaded model:", sample_prediction)


import joblib
import pandas as pd

# ==============================
# 1. LOAD SAVED MODEL
# ==============================
model = joblib.load("amazon_rating_prediction_model.pkl")
print("Model loaded successfully")

# ==============================
# 2. TAKE NEW INPUT DATA
# (Example: data coming from frontend or user)
# ==============================

new_data = {
    'discounted_price': [999],
    'actual_price': [1999],
    'discount_percentage': [50],
    'rating_count': [12000]
}

# Convert input to DataFrame
input_df = pd.DataFrame(new_data)

print("\nInput Data:")
print(input_df)

# ==============================
# 3. MAKE PREDICTION
# ==============================

predicted_rating = model.predict(input_df)

print("\nPredicted Rating:", predicted_rating[0])



import matplotlib.pyplot as plt
import numpy as np

# ==============================
# 1. SAMPLE ACTUAL & PREDICTED VALUES
# (In real case, these come from test data)
# ==============================

actual_sales = y_test.values
predicted_sales = y_pred

# ==============================
# 2. SCATTER PLOT (Actual vs Predicted)
# ==============================

plt.figure()
plt.scatter(actual_sales, predicted_sales)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# ==============================
# 3. LINE GRAPH (Comparison Trend)
# ==============================

plt.figure()
plt.plot(actual_sales[:50], label="Actual Sales")
plt.plot(predicted_sales[:50], label="Predicted Sales")
plt.xlabel("Sample Index")
plt.ylabel("Sales Value")
plt.title("Actual vs Predicted Sales Trend")
plt.legend()
plt.show()


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

