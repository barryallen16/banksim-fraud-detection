import gradio as gr
import pandas as pd
import numpy as np
import joblib
# Required for deserializing preprocessor.pkl
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import DBSCAN  # Use cuml.cluster.DBSCAN if GPU available

# Load the saved preprocessor and model
try:
    preprocessor = joblib.load('preprocessor.pkl')
    dbscan = joblib.load('dbscan_fraud_model.pkl')
except FileNotFoundError as e:
    raise FileNotFoundError("Error: Ensure 'preprocessor.pkl' and 'dbscan_fraud_model.pkl' are in the working directory.") from e
except Exception as e:
    raise Exception(f"Error loading model or preprocessor: {str(e)}") from e

# Define feature engineering function (same as in your notebook)
def add_features(df):
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['month'] = df['trans_date_trans_time'].dt.month
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c
    
    df['distance'] = haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    df['unix_time'] = df['trans_date_trans_time'].astype(np.int64) // 10**9
    return df

# Prediction function
def predict_fraud(amt, category, city_pop, lat, long, merch_lat, merch_long, trans_date_trans_time):
    try:
        # Create DataFrame from inputs
        data = pd.DataFrame({
            'amt': [float(amt)],
            'category': [category],
            'city_pop': [int(city_pop)],
            'lat': [float(lat)],
            'long': [float(long)],
            'merch_lat': [float(merch_lat)],
            'merch_long': [float(merch_long)],
            'trans_date_trans_time': [trans_date_trans_time]
        })
        
        # Add features
        data = add_features(data)
        
        # Define columns
        numerical_cols = ['amt', 'city_pop', 'unix_time', 'distance', 'hour', 'day_of_week', 'month']
        categorical_cols = ['category']
        
        # Preprocess input
        X = data[numerical_cols + categorical_cols]
        X_preprocessed = preprocessor.transform(X)
        
        # Predict using fit_predict
        cluster = dbscan.fit_predict(X_preprocessed)
        is_anomaly = cluster == -1
        
        return "Fraud" if is_anomaly[0] else "Not Fraud"
    except ValueError as e:
        return f"Error: Invalid input. Ensure all fields are correctly filled (e.g., valid date format, numerical values). Details: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

# Define Gradio interface
demo = gr.Interface(
    fn=predict_fraud,
    inputs=[
        gr.Number(label="Transaction Amount ($)", value=100.0, precision=2),
        gr.Dropdown(
            choices=[
                "personal_care", "health_fitness", "misc_pos", "travel", "kids_pets",
                "shopping_pos", "food_dining", "home", "entertainment", "shopping_net",
                "misc_net", "grocery_pos", "gas_transport", "grocery_net"
            ],
            label="Transaction Category",
            value="personal_care"
        ),
        gr.Number(label="City Population", value=10000, precision=0),
        gr.Number(label="Customer Latitude", value=40.0, precision=4),
        gr.Number(label="Customer Longitude", value=-74.0, precision=4),
        gr.Number(label="Merchant Latitude", value=40.0, precision=4),
        gr.Number(label="Merchant Longitude", value=-74.0, precision=4),
        gr.Textbox(label="Transaction Date and Time (YYYY-MM-DD HH:MM:SS)", value="2025-06-14 12:00:00")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Fraud Detection Demo",
    description="Enter transaction details to predict if it's fraudulent.",
    theme=gr.themes.Soft()
)

# Launch the interface
if __name__ == "__main__":
    demo.launch()