import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class CarDataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def preprocess_data(self, df):
        """
        Preprocess the car dataset by encoding categorical variables and scaling numerical features.
        """
        # Create a copy to avoid modifying the original dataframe
        processed_df = df.copy()

        # Function to convert price to numerical value
        def convert_price(price_str):
            # Remove commas from the string for conversion
            price_str = price_str.replace(',', '').strip()
            
            # Check for 'Crore' in the string
            if 'Crore' in price_str:
                return float(price_str.replace(' Crore', '').strip()) * 100  # Convert Crore to Lakh
            # Check for 'Lakh' in the string
            elif 'Lakh' in price_str:
                return float(price_str.replace(' Lakh', '').strip())
            # Check for plain numerical values (e.g., '99000')
            elif price_str.isdigit():  # Check if it is purely numeric
                return float(price_str) / 100000  # Convert to Lakh
            else:
                raise ValueError(f"Unexpected price format: {price_str}")
        
        # Convert price from 'Lakh' or 'Crore' to numerical value
        processed_df['car_prices_in_rupee'] = processed_df['car_prices_in_rupee'].apply(convert_price)
        
        # Clean and convert kms_driven
        processed_df['kms_driven'] = processed_df['kms_driven'].str.replace(' kms', '').str.replace(',', '').astype(float)
        
        # Encode categorical variables
        categorical_columns = ['fuel_type', 'transmission', 'ownership', 'engine']
        for column in categorical_columns:
            self.label_encoders[column] = LabelEncoder()
            processed_df[column] = self.label_encoders[column].fit_transform(processed_df[column])
        
        # Convert manufacture year to age
        current_year = 2024
        processed_df['age'] = current_year - processed_df['manufacture']
        
        # Select features for model
        features = ['kms_driven', 'age'] + categorical_columns
        
        # Scale numerical features
        X = self.scaler.fit_transform(processed_df[features])
        y = processed_df['car_prices_in_rupee'].values
        
        return X, y, features
    
    def prepare_data(self, df, test_size=0.2, random_state=42):
        """
        Prepare data by splitting into train and test sets.
        """
        X, y, features = self.preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test, features
