import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.layers import (LSTM, BatchNormalization, Concatenate,
                                     Conv1D, Dense, Dropout, Flatten,
                                     GlobalAveragePooling1D, Input,
                                     LayerNormalization, MaxPooling1D,
                                     MultiHeadAttention)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

class WeatherPredictor:
    def __init__(self, sequence_length=168, prediction_horizon=24):
        """
        Advanced Weather Prediction Model
        
        Args:
            sequence_length: Number of hours to look back (default: 168 = 7 days)
            prediction_horizon: Number of hours to predict ahead (default: 24 = 1 day)
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scalers = {}
        self.model = None
        self.history = None
        
    def create_sample_data(self, n_samples=10000, save_to_file=True):
        """
        Create sample weather data with realistic patterns
        """
        np.random.seed(42)
        
        # Generate datetime index
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(hours=i) for i in range(n_samples)]
        
        # Generate realistic weather patterns
        data = []
        
        for i, date in enumerate(dates):
            # Seasonal temperature patterns
            day_of_year = date.timetuple().tm_yday
            seasonal_temp = 20 + 15 * np.sin(2 * np.pi * day_of_year / 365.25)
            
            # Daily temperature cycle  
            hour_of_day = date.hour
            daily_temp_cycle = 5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
            
            # Random variations
            temp_noise = np.random.normal(0, 3)
            temperature = seasonal_temp + daily_temp_cycle + temp_noise
            
            # Humidity (inversely related to temperature with noise)
            humidity = max(10, min(100, 80 - 0.8 * (temperature - 15) + np.random.normal(0, 10)))
            
            # Pressure (realistic atmospheric pressure)
            pressure = 1013.25 + np.random.normal(0, 15) + 5 * np.sin(2 * np.pi * i / (24 * 7))
            
            # Wind speed (more variable)
            wind_speed = max(0, np.random.exponential(5) + 2 * np.sin(2 * np.pi * i / (24 * 3)))
            
            # Wind direction (0-360 degrees)
            wind_direction = (np.random.normal(180, 60) + 30 * np.sin(2 * np.pi * i / (24 * 2))) % 360
            
            # Precipitation (correlated with humidity and pressure)
            precip_prob = max(0, (humidity - 60) / 40 + (1013.25 - pressure) / 50)
            precipitation = np.random.exponential(2) if np.random.random() < precip_prob * 0.1 else 0
            
            # Cloud cover (related to humidity and precipitation)
            cloud_cover = min(100, max(0, humidity * 0.8 + precipitation * 10 + np.random.normal(0, 15)))
            
            # UV Index (related to cloud cover and time of day)
            if 6 <= hour_of_day <= 18:  # Daylight hours
                uv_index = max(0, 8 * (1 - cloud_cover/100) * np.sin(np.pi * (hour_of_day - 6) / 12))
            else:
                uv_index = 0
            
            # Visibility (inversely related to precipitation and humidity)
            visibility = max(0.1, 25 - precipitation * 2 - (humidity - 50) * 0.1 + np.random.normal(0, 2))
            
            # Location data (example coordinates)
            latitude = 40.7128 + np.random.normal(0, 10)  # Around NYC with variation
            longitude = -74.0060 + np.random.normal(0, 10)
            elevation = max(0, np.random.exponential(100))
            
            # Month and hour for cyclical encoding
            month = date.month
            hour = date.hour
            day_of_week = date.weekday()
            
            data.append({
                'datetime': date,
                'temperature': round(temperature, 2),
                'humidity': round(humidity, 2),
                'pressure': round(pressure, 2),
                'wind_speed': round(wind_speed, 2),
                'wind_direction': round(wind_direction, 2),
                'precipitation': round(precipitation, 2),
                'cloud_cover': round(cloud_cover, 2),
                'uv_index': round(uv_index, 2),
                'visibility': round(visibility, 2),
                'latitude': round(latitude, 4),
                'longitude': round(longitude, 4),
                'elevation': round(elevation, 2),
                'month': month,
                'hour': hour,
                'day_of_week': day_of_week
            })
        
        df = pd.DataFrame(data)
        
        if save_to_file:
            df.to_csv('weather_data_sample.csv', index=False)
            print("Sample data saved to 'weather_data_sample.csv'")
            print("\nData format:")
            print(df.head())
            print(f"\nDataset shape: {df.shape}")
            print("\nColumns description:")
            print("- datetime: Timestamp of observation")
            print("- temperature: Temperature in Celsius")
            print("- humidity: Relative humidity (%)")
            print("- pressure: Atmospheric pressure (hPa)")
            print("- wind_speed: Wind speed (m/s)")
            print("- wind_direction: Wind direction (degrees)")
            print("- precipitation: Precipitation amount (mm)")
            print("- cloud_cover: Cloud cover percentage")
            print("- uv_index: UV index")
            print("- visibility: Visibility (km)")
            print("- latitude/longitude: Geographic coordinates")
            print("- elevation: Elevation above sea level (m)")
            print("- month/hour/day_of_week: Time features")
        
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess the weather data for training
        """
        print("Preprocessing data...")
        
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Create cyclical features for time components
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Wind direction cyclical encoding
        df['wind_dir_sin'] = np.sin(2 * np.pi * df['wind_direction'] / 360)
        df['wind_dir_cos'] = np.cos(2 * np.pi * df['wind_direction'] / 360)
        
        # Feature columns for training
        feature_columns = [
            'temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation',
            'cloud_cover', 'uv_index', 'visibility', 'latitude', 'longitude', 
            'elevation', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
            'day_sin', 'day_cos', 'wind_dir_sin', 'wind_dir_cos'
        ]
        
        # Target columns (what we want to predict)
        target_columns = ['temperature', 'humidity', 'precipitation', 'wind_speed']
        
        # Scale features
        for col in feature_columns:
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])
            self.scalers[col] = scaler
        
        return df[feature_columns], df[target_columns], feature_columns, target_columns
    
    def create_sequences(self, features, targets):
        """
        Create sequences for time-series prediction
        """
        print(f"Creating sequences with length {self.sequence_length}...")
        
        X, y = [], []
        
        for i in range(len(features) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            X.append(features.iloc[i:(i + self.sequence_length)].values)
            
            # Target sequence (next prediction_horizon hours)
            target_start = i + self.sequence_length
            target_end = target_start + self.prediction_horizon
            y.append(targets.iloc[target_start:target_end].values)
        
        return np.array(X), np.array(y)
    
    def build_advanced_model(self, input_shape, output_shape):
        """
        Build an advanced hybrid neural network for weather prediction
        Combines LSTM, CNN, and Attention mechanisms
        """
        print("Building advanced neural network model...")
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # CNN branch for pattern recognition
        cnn_branch = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        cnn_branch = BatchNormalization()(cnn_branch)
        cnn_branch = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(cnn_branch)
        cnn_branch = BatchNormalization()(cnn_branch)
        cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
        cnn_branch = Dropout(0.2)(cnn_branch)
        
        # LSTM branch for temporal dependencies
        lstm_branch = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
        lstm_branch = BatchNormalization()(lstm_branch)
        lstm_branch = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(lstm_branch)
        lstm_branch = BatchNormalization()(lstm_branch)
        
        # Attention mechanism
        attention = MultiHeadAttention(num_heads=8, key_dim=64)(lstm_branch, lstm_branch)
        attention = LayerNormalization()(attention + lstm_branch)
        attention = GlobalAveragePooling1D()(attention)
        
        # Combine CNN and LSTM-Attention branches
        cnn_flat = GlobalAveragePooling1D()(cnn_branch)
        combined = Concatenate()([cnn_flat, attention])
        
        # Dense layers for final prediction
        dense = Dense(256, activation='relu')(combined)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.3)(dense)
        
        dense = Dense(128, activation='relu')(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.2)(dense)
        
        dense = Dense(64, activation='relu')(dense)
        dense = Dropout(0.1)(dense)
        
        # Output layer - reshape to match target shape
        output = Dense(np.prod(output_shape), activation='linear')(dense)
        output = tf.keras.layers.Reshape(output_shape)(output)
        
        model = Model(inputs=inputs, outputs=output)
        
        # Compile with advanced optimizer
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        model.compile(
            optimizer=optimizer,
            loss='huber',  # Robust to outliers
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train_model(self, df=None, csv_file=None, validation_split=0.2, epochs=100, batch_size=32):
        """
        Train the weather prediction model
        """
        print("Starting model training...")
        
        # Load data if csv_file is provided
        if csv_file is not None:
            print(f"Loading data from {csv_file}...")
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} records from CSV file")
        elif df is None:
            raise ValueError("Either df or csv_file must be provided")
        
        # Preprocess data
        features, targets, feature_cols, target_cols = self.preprocess_data(df)
        
        # Create sequences
        X, y = self.create_sequences(features, targets)
        
        print(f"Sequence shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, shuffle=False
        )
        
        # Build model
        self.model = self.build_advanced_model(
            input_shape=(X.shape[1], X.shape[2]),
            output_shape=(y.shape[1], y.shape[2])
        )
        
        print(f"Model built with {self.model.count_params():,} parameters")
        print(self.model.summary())
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'best_weather_model.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        train_loss = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss = self.model.evaluate(X_val, y_val, verbose=0)
        
        print(f"\nTraining completed!")
        print(f"Final Training Loss: {train_loss[0]:.4f}")
        print(f"Final Validation Loss: {val_loss[0]:.4f}")
        
        return self.history
    
    def predict_weather(self, input_sequence):
        """
        Predict weather for the next prediction_horizon hours
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Reshape input if needed
        if len(input_sequence.shape) == 2:
            input_sequence = input_sequence.reshape(1, input_sequence.shape[0], input_sequence.shape[1])
        
        prediction = self.model.predict(input_sequence, verbose=0)
        return prediction
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # MAE
        axes[0, 1].plot(self.history.history['mae'], label='Training MAE')
        axes[0, 1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        
        # MSE
        axes[1, 0].plot(self.history.history['mse'], label='Training MSE')
        axes[1, 0].plot(self.history.history['val_mse'], label='Validation MSE')
        axes[1, 0].set_title('Mean Squared Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].legend()
        
        # Learning Rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('LR')
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath='weather_predictor_model.keras'):
        """
        Save the trained model
        """
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save.")
    
    def load_model(self, filepath='weather_predictor_model.keras'):
        """
        Load a trained model
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

# Example usage and demonstration
def main():
    """
    Main function to demonstrate the weather prediction model
    """
    print("🌤️  Advanced Weather Prediction Neural Network")
    print("=" * 50)
    
    # Initialize predictor
    predictor = WeatherPredictor(sequence_length=168, prediction_horizon=24)
    
    # Check if the existing CSV file exists
    csv_filename = 'hourly_weather_data_in_new_york_city-1754407915243.csv'
    
    try:
        # Try to use existing CSV file
        print(f"\n1. Loading existing weather dataset: {csv_filename}")
        
        # Train model with existing data
        print("\n2. Training the neural network...")
        history = predictor.train_model(
            csv_file=csv_filename,
            validation_split=0.2, 
            epochs=50,  # Reduce for demonstration
            batch_size=64
        )
        
    except FileNotFoundError:
        print(f"CSV file {csv_filename} not found. Creating sample data...")
        # Create sample data as fallback
        sample_data = predictor.create_sample_data(n_samples=8760)  # 1 year of hourly data
        
        # Train model
        print("\n2. Training the neural network...")
        history = predictor.train_model(
            df=sample_data, 
            validation_split=0.2, 
            epochs=50,  # Reduce for demonstration
            batch_size=64
        )
    
    # Plot training history
    print("\n3. Plotting training results...")
    predictor.plot_training_history()
    
    # Save model
    print("\n4. Saving trained model...")
    predictor.save_model('advanced_weather_model.keras')
    
    print("\n✅ Weather prediction model training completed!")
    print("\nModel capabilities:")
    print("- Predicts temperature, humidity, precipitation, and wind speed")
    print("- Uses 7 days of historical data to predict next 24 hours")
    print("- Combines LSTM, CNN, and Attention mechanisms")
    print("- Handles multiple weather variables simultaneously")
    print("- Includes geographical and temporal features")
    
    return predictor

if __name__ == "__main__":
    # Run the demonstration
    weather_model = main()
if __name__ == "__main__":
    # Run the demonstration
    weather_model = main()