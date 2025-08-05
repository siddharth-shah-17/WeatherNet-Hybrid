"""
Weather Prediction Model Inference Module

This module provides a production-ready interface for making weather predictions
using the trained hybrid CNN-LSTM-Attention model.

Author: Advanced Weather Prediction Team
Created: 2024
Version: 1.0
"""

import json
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

class WeatherInference:
    """
    Production-ready weather prediction inference class.
    
    This class provides methods for loading trained models and making
    weather predictions with proper data preprocessing and postprocessing.
    """
    
    def __init__(self, model_path: str = 'models/advanced_weather_model.keras'):
        """
        Initialize the Weather Inference system.
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.scalers = {}
        self.sequence_length = 168  # 7 days
        self.prediction_horizon = 24  # 24 hours
        self.target_columns = ['temperature', 'humidity', 'precipitation', 'wind_speed']
        self.feature_columns = [
            'temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation',
            'cloud_cover', 'uv_index', 'visibility', 'latitude', 'longitude', 
            'elevation', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
            'day_sin', 'day_cos', 'wind_dir_sin', 'wind_dir_cos'
        ]
        
    def load_model(self) -> bool:
        """
        Load the trained weather prediction model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                return False
                
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded successfully: {self.model_path}")
            print(f"📊 Model parameters: {self.model.count_params():,}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def _create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cyclical features for temporal variables.
        
        Args:
            df (pd.DataFrame): Input dataframe with datetime information
            
        Returns:
            pd.DataFrame: Dataframe with cyclical features added
        """
        df = df.copy()
        
        # Extract time components if datetime column exists
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['month'] = df['datetime'].dt.month
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
        
        # Create cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Wind direction cyclical encoding
        if 'wind_direction' in df.columns:
            df['wind_dir_sin'] = np.sin(2 * np.pi * df['wind_direction'] / 360)
            df['wind_dir_cos'] = np.cos(2 * np.pi * df['wind_direction'] / 360)
        else:
            df['wind_dir_sin'] = 0
            df['wind_dir_cos'] = 1
        
        return df
    
    def _prepare_scalers(self, df: pd.DataFrame) -> None:
        """
        Prepare scalers for feature normalization.
        
        Args:
            df (pd.DataFrame): Training data to fit scalers
        """
        for col in self.feature_columns:
            if col in df.columns:
                scaler = MinMaxScaler()
                scaler.fit(df[[col]])
                self.scalers[col] = scaler
    
    def preprocess_data(self, df: pd.DataFrame, fit_scalers: bool = False) -> np.ndarray:
        """
        Preprocess input data for model inference.
        
        Args:
            df (pd.DataFrame): Input weather data
            fit_scalers (bool): Whether to fit new scalers or use existing ones
            
        Returns:
            np.ndarray: Preprocessed feature array
        """
        df = df.copy()
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Create cyclical features
        df = self._create_cyclical_features(df)
        
        # Fit scalers if needed
        if fit_scalers or not self.scalers:
            self._prepare_scalers(df)
        
        # Scale features
        for col in self.feature_columns:
            if col in df.columns and col in self.scalers:
                df[col] = self.scalers[col].transform(df[[col]]).flatten()
            elif col not in df.columns:
                # Fill missing columns with defaults
                df[col] = 0.5  # Neutral scaled value
        
        return df[self.feature_columns].values
    
    def predict_single_sequence(self, 
                              input_data: Union[pd.DataFrame, np.ndarray],
                              return_confidence: bool = False) -> Dict:
        """
        Make weather predictions for a single input sequence.
        
        Args:
            input_data: Input weather data (last 168 hours)
            return_confidence: Whether to return prediction confidence metrics
            
        Returns:
            Dict: Prediction results with timestamps and values
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess input data
        if isinstance(input_data, pd.DataFrame):
            if len(input_data) < self.sequence_length:
                raise ValueError(f"Need at least {self.sequence_length} hours of data")
            
            # Take the last sequence_length records
            input_data = input_data.tail(self.sequence_length)
            features = self.preprocess_data(input_data)
        else:
            features = input_data
        
        # Reshape for model input
        if len(features.shape) == 2:
            features = features.reshape(1, features.shape[0], features.shape[1])
        
        # Make prediction
        prediction = self.model.predict(features, verbose=0)[0]
        
        # Create results dictionary
        results = {
            'predictions': {},
            'metadata': {
                'model_path': self.model_path,
                'prediction_horizon': self.prediction_horizon,
                'timestamp': datetime.now().isoformat(),
                'input_sequence_length': self.sequence_length
            }
        }
        
        # Process predictions for each variable
        for i, var in enumerate(self.target_columns):
            var_predictions = prediction[:, i].tolist()
            
            results['predictions'][var] = {
                'values': var_predictions,
                'hours_ahead': list(range(1, self.prediction_horizon + 1)),
                'unit': self._get_unit(var)
            }
            
            if return_confidence:
                # Simple confidence metric based on prediction variance
                confidence = max(0, 1 - np.std(var_predictions))
                results['predictions'][var]['confidence'] = confidence
        
        return results
    
    def predict_batch(self, 
                     input_sequences: List[Union[pd.DataFrame, np.ndarray]]) -> List[Dict]:
        """
        Make predictions for multiple input sequences.
        
        Args:
            input_sequences: List of input sequences
            
        Returns:
            List[Dict]: List of prediction results
        """
        results = []
        
        for i, sequence in enumerate(input_sequences):
            try:
                result = self.predict_single_sequence(sequence)
                result['metadata']['sequence_id'] = i
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing sequence {i}: {e}")
                results.append({
                    'error': str(e),
                    'sequence_id': i
                })
        
        return results
    
    def predict_from_csv(self, 
                        csv_file: str, 
                        output_file: Optional[str] = None) -> Dict:
        """
        Make predictions from CSV file input.
        
        Args:
            csv_file: Path to input CSV file
            output_file: Optional path to save results
            
        Returns:
            Dict: Prediction results
        """
        try:
            # Load data
            df = pd.read_csv(csv_file)
            print(f"📊 Loaded {len(df)} records from {csv_file}")
            
            # Make prediction
            results = self.predict_single_sequence(df, return_confidence=True)
            
            # Save results if output file specified
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"💾 Results saved to {output_file}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error processing CSV file: {e}")
            return {'error': str(e)}
    
    def _get_unit(self, variable: str) -> str:
        """Get the unit for a weather variable."""
        units = {
            'temperature': '°C',
            'humidity': '%',
            'precipitation': 'mm',
            'wind_speed': 'm/s'
        }
        return units.get(variable, '')
    
    def create_forecast_summary(self, prediction_results: Dict) -> str:
        """
        Create a human-readable forecast summary.
        
        Args:
            prediction_results: Results from predict_single_sequence
            
        Returns:
            str: Formatted forecast summary
        """
        if 'error' in prediction_results:
            return f"Error in prediction: {prediction_results['error']}"
        
        predictions = prediction_results['predictions']
        summary_lines = []
        
        summary_lines.append("🌤️ 24-Hour Weather Forecast")
        summary_lines.append("=" * 40)
        
        # Get average values for next 24 hours
        for var in self.target_columns:
            if var in predictions:
                values = predictions[var]['values']
                unit = predictions[var]['unit']
                avg_val = np.mean(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                var_name = var.replace('_', ' ').title()
                summary_lines.append(
                    f"{var_name:12}: {avg_val:.1f}{unit} "
                    f"(Range: {min_val:.1f}-{max_val:.1f}{unit})"
                )
        
        # Add confidence if available
        if 'confidence' in predictions.get('temperature', {}):
            avg_confidence = np.mean([
                predictions[var].get('confidence', 0) 
                for var in self.target_columns if var in predictions
            ])
            summary_lines.append(f"\nOverall Confidence: {avg_confidence:.1%}")
        
        summary_lines.append(f"\nGenerated: {prediction_results['metadata']['timestamp']}")
        
        return '\n'.join(summary_lines)

def main():
    """
    Demonstration of the inference system.
    """
    print("🌤️ Weather Prediction Inference System")
    print("=" * 50)
    
    # Initialize inference system
    inference = WeatherInference()
    
    # Load model
    if not inference.load_model():
        print("❌ Failed to load model. Please check the model path.")
        return
    
    # Example: Predict from CSV file
    csv_file = 'data/weather_data_nyc.csv'
    
    if os.path.exists(csv_file):
        print(f"\n📊 Making prediction from {csv_file}")
        results = inference.predict_from_csv(csv_file, 'results/forecast_output.json')
        
        if 'error' not in results:
            # Print forecast summary
            summary = inference.create_forecast_summary(results)
            print(f"\n{summary}")
            
            # Print detailed predictions
            print(f"\n📈 Detailed 24-Hour Forecast:")
            predictions = results['predictions']
            
            for hour in range(1, min(6, 25)):  # Show first 5 hours
                print(f"\nHour +{hour}:")
                for var in inference.target_columns:
                    if var in predictions:
                        val = predictions[var]['values'][hour-1]
                        unit = predictions[var]['unit']
                        var_name = var.replace('_', ' ').title()
                        print(f"  {var_name:12}: {val:.2f}{unit}")
        else:
            print(f"❌ Prediction failed: {results['error']}")
    else:
        print(f"❌ Data file not found: {csv_file}")
        print("Please ensure you have weather data in the expected location.")

if __name__ == "__main__":
    main()
