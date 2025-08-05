import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

class AdvancedWeatherTester:
    def __init__(self, model_path='advanced_weather_model.keras'):
        self.model_path = model_path
        self.model = None
        self.scalers = {}
        self.sequence_length = 168
        self.prediction_horizon = 24
        self.target_columns = ['temperature', 'humidity', 'precipitation', 'wind_speed']
        
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded: {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def preprocess_data(self, df):
        """Preprocess data with same scaling as training"""
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Create cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['wind_dir_sin'] = np.sin(2 * np.pi * df['wind_direction'] / 360)
        df['wind_dir_cos'] = np.cos(2 * np.pi * df['wind_direction'] / 360)
        
        feature_columns = [
            'temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation',
            'cloud_cover', 'uv_index', 'visibility', 'latitude', 'longitude', 
            'elevation', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
            'day_sin', 'day_cos', 'wind_dir_sin', 'wind_dir_cos'
        ]
        
        # Scale features
        for col in feature_columns:
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])
            self.scalers[col] = scaler
        
        return df[feature_columns], df[self.target_columns]
    
    def evaluate_model_accuracy(self, csv_file='hourly_weather_data_in_new_york_city-1754407915243.csv'):
        """Comprehensive model accuracy evaluation"""
        print("🔍 Comprehensive Model Accuracy Evaluation")
        print("=" * 60)
        
        # Load and preprocess data
        df = pd.read_csv(csv_file)
        features, targets = self.preprocess_data(df.copy())
        
        # Create multiple test sequences
        predictions_all = []
        actuals_all = []
        
        # Test on last 30 sequences (30 different 7-day periods)
        start_indices = range(len(features) - self.sequence_length - self.prediction_horizon - 30, 
                            len(features) - self.sequence_length - self.prediction_horizon, 1)
        
        print(f"Testing on {len(start_indices)} different time periods...")
        
        for i, start_idx in enumerate(start_indices):
            if start_idx < 0:
                continue
                
            # Create input sequence
            input_seq = features.iloc[start_idx:start_idx + self.sequence_length].values
            input_seq = input_seq.reshape(1, self.sequence_length, -1)
            
            # Get actual target values
            actual_start = start_idx + self.sequence_length
            actual_end = actual_start + self.prediction_horizon
            actual_values = targets.iloc[actual_start:actual_end].values
            
            # Make prediction
            prediction = self.model.predict(input_seq, verbose=0)[0]
            
            predictions_all.append(prediction)
            actuals_all.append(actual_values)
        
        # Convert to numpy arrays
        predictions_all = np.array(predictions_all)  # Shape: (n_tests, 24, 4)
        actuals_all = np.array(actuals_all)
        
        # Calculate metrics for each weather variable
        metrics = {}
        
        print("\n📊 Accuracy Metrics by Weather Variable:")
        print("-" * 50)
        
        for i, var in enumerate(self.target_columns):
            pred_var = predictions_all[:, :, i].flatten()
            actual_var = actuals_all[:, :, i].flatten()
            
            # Calculate metrics
            mse = mean_squared_error(actual_var, pred_var)
            mae = mean_absolute_error(actual_var, pred_var)
            rmse = np.sqrt(mse)
            r2 = r2_score(actual_var, pred_var)
            
            # Calculate percentage accuracy
            mape = np.mean(np.abs((actual_var - pred_var) / (actual_var + 1e-8))) * 100
            
            metrics[var] = {
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape
            }
            
            print(f"{var.upper():15} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f} | MAPE: {mape:.1f}%")
        
        return predictions_all, actuals_all, metrics
    
    def create_accuracy_visualization(self, predictions, actuals, metrics):
        """Create detailed accuracy visualizations"""
        print("\n📈 Creating detailed accuracy visualizations...")
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Scatter plots: Predicted vs Actual
        for i, var in enumerate(self.target_columns):
            plt.subplot(4, 3, i*3 + 1)
            
            pred_flat = predictions[:, :, i].flatten()
            actual_flat = actuals[:, :, i].flatten()
            
            plt.scatter(actual_flat, pred_flat, alpha=0.5, s=10)
            
            # Perfect prediction line
            min_val, max_val = min(actual_flat.min(), pred_flat.min()), max(actual_flat.max(), pred_flat.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            plt.xlabel(f'Actual {var.replace("_", " ").title()}')
            plt.ylabel(f'Predicted {var.replace("_", " ").title()}')
            plt.title(f'{var.replace("_", " ").title()}: Predicted vs Actual\nR² = {metrics[var]["R²"]:.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 2. Error distribution histograms
        for i, var in enumerate(self.target_columns):
            plt.subplot(4, 3, i*3 + 2)
            
            pred_flat = predictions[:, :, i].flatten()
            actual_flat = actuals[:, :, i].flatten()
            errors = pred_flat - actual_flat
            
            plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
            plt.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean Error: {errors.mean():.3f}')
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.title(f'{var.replace("_", " ").title()}: Error Distribution\nMAE = {metrics[var]["MAE"]:.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 3. Time series comparison (sample)
        for i, var in enumerate(self.target_columns):
            plt.subplot(4, 3, i*3 + 3)
            
            # Show first test case as example
            sample_pred = predictions[0, :, i]
            sample_actual = actuals[0, :, i]
            hours = range(1, 25)
            
            plt.plot(hours, sample_actual, 'o-', label='Actual', linewidth=2, markersize=4)
            plt.plot(hours, sample_pred, 's-', label='Predicted', linewidth=2, markersize=4)
            
            plt.xlabel('Hours Ahead')
            plt.ylabel(var.replace("_", " ").title())
            plt.title(f'{var.replace("_", " ").title()}: 24h Forecast Sample')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('🌤️ Comprehensive Model Accuracy Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.show()
    
    def test_seasonal_performance(self, csv_file='hourly_weather_data_in_new_york_city-1754407915243.csv'):
        """Test model performance across different seasons"""
        print("\n🌸🌞🍂❄️ Seasonal Performance Analysis")
        print("=" * 50)
        
        df = pd.read_csv(csv_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Define seasons
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        df['season'] = df['month'].apply(get_season)
        
        # Test performance by season
        seasonal_results = {}
        
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            season_data = df[df['season'] == season].copy()
            
            if len(season_data) < self.sequence_length + self.prediction_horizon:
                continue
            
            print(f"\n{season} Analysis ({len(season_data)} records):")
            
            features, targets = self.preprocess_data(season_data)
            
            # Test on a few sequences from this season
            test_sequences = min(5, len(features) - self.sequence_length - self.prediction_horizon)
            
            if test_sequences <= 0:
                continue
            
            season_predictions = []
            season_actuals = []
            
            for i in range(test_sequences):
                start_idx = i * 10  # Spread out the test points
                
                if start_idx + self.sequence_length + self.prediction_horizon > len(features):
                    break
                
                input_seq = features.iloc[start_idx:start_idx + self.sequence_length].values
                input_seq = input_seq.reshape(1, self.sequence_length, -1)
                
                actual_start = start_idx + self.sequence_length
                actual_end = actual_start + self.prediction_horizon
                actual_values = targets.iloc[actual_start:actual_end].values
                
                prediction = self.model.predict(input_seq, verbose=0)[0]
                
                season_predictions.append(prediction)
                season_actuals.append(actual_values)
            
            if season_predictions:
                season_predictions = np.array(season_predictions)
                season_actuals = np.array(season_actuals)
                
                # Calculate seasonal metrics
                seasonal_metrics = {}
                for i, var in enumerate(self.target_columns):
                    pred_var = season_predictions[:, :, i].flatten()
                    actual_var = season_actuals[:, :, i].flatten()
                    
                    mae = mean_absolute_error(actual_var, pred_var)
                    rmse = np.sqrt(mean_squared_error(actual_var, pred_var))
                    r2 = r2_score(actual_var, pred_var)
                    
                    seasonal_metrics[var] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
                    print(f"  {var:12}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")
                
                seasonal_results[season] = seasonal_metrics
        
        return seasonal_results
    
    def create_prediction_confidence_analysis(self, csv_file='hourly_weather_data_in_new_york_city-1754407915243.csv'):
        """Analyze prediction confidence across different forecast horizons"""
        print("\n🎯 Prediction Confidence Analysis (1-24 hours ahead)")
        print("=" * 60)
        
        df = pd.read_csv(csv_file)
        features, targets = self.preprocess_data(df.copy())
        
        # Test on multiple sequences
        n_tests = 20
        start_indices = range(len(features) - self.sequence_length - self.prediction_horizon - n_tests, 
                            len(features) - self.sequence_length - self.prediction_horizon, 1)
        
        hourly_errors = {var: [] for var in self.target_columns}
        
        for start_idx in start_indices:
            input_seq = features.iloc[start_idx:start_idx + self.sequence_length].values
            input_seq = input_seq.reshape(1, self.sequence_length, -1)
            
            actual_start = start_idx + self.sequence_length
            actual_end = actual_start + self.prediction_horizon
            actual_values = targets.iloc[actual_start:actual_end].values
            
            prediction = self.model.predict(input_seq, verbose=0)[0]
            
            # Calculate error for each hour ahead
            for hour in range(self.prediction_horizon):
                for i, var in enumerate(self.target_columns):
                    error = abs(prediction[hour, i] - actual_values[hour, i])
                    hourly_errors[var].append((hour + 1, error))
        
        # Plot confidence analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, var in enumerate(self.target_columns):
            ax = axes[i]
            
            # Group errors by hour
            hours = list(range(1, 25))
            hour_mae = []
            
            for hour in hours:
                hour_errors = [error for h, error in hourly_errors[var] if h == hour]
                hour_mae.append(np.mean(hour_errors))
            
            ax.plot(hours, hour_mae, 'o-', linewidth=2, markersize=6)
            ax.set_xlabel('Hours Ahead')
            ax.set_ylabel('Mean Absolute Error')
            ax.set_title(f'{var.replace("_", " ").title()}: Prediction Accuracy vs Forecast Horizon')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(hours, hour_mae, 1)
            p = np.poly1d(z)
            ax.plot(hours, p(hours), "--", alpha=0.8, color='red', 
                   label=f'Trend: {"Increasing" if z[0] > 0 else "Stable"} error')
            ax.legend()
        
        plt.tight_layout()
        plt.suptitle('🔮 Model Confidence: Error vs Forecast Horizon', fontsize=14, fontweight='bold', y=0.98)
        plt.show()
        
        # Print summary
        print("\nConfidence Summary:")
        for var in self.target_columns:
            hour_1_errors = [error for h, error in hourly_errors[var] if h == 1]
            hour_24_errors = [error for h, error in hourly_errors[var] if h == 24]
            
            mae_1h = np.mean(hour_1_errors)
            mae_24h = np.mean(hour_24_errors)
            
            confidence_change = ((mae_24h - mae_1h) / mae_1h) * 100
            
            print(f"  {var:15}: 1h MAE={mae_1h:.3f}, 24h MAE={mae_24h:.3f} "
                  f"({confidence_change:+.1f}% change)")
    
    def run_comprehensive_analysis(self):
        """Run complete model analysis"""
        print("🚀 Advanced Weather Model Analysis")
        print("=" * 60)
        
        if not self.load_model():
            return
        
        # 1. Overall accuracy evaluation
        predictions, actuals, metrics = self.evaluate_model_accuracy()
        
        # 2. Create accuracy visualizations
        self.create_accuracy_visualization(predictions, actuals, metrics)
        
        # 3. Seasonal performance analysis
        seasonal_results = self.test_seasonal_performance()
        
        # 4. Prediction confidence analysis
        self.create_prediction_confidence_analysis()
        
        print("\n🎉 Comprehensive Analysis Complete!")
        print("\n📋 Model Performance Summary:")
        print("-" * 40)
        
        for var in self.target_columns:
            r2 = metrics[var]['R²']
            mae = metrics[var]['MAE']
            
            if r2 > 0.8:
                performance = "Excellent ⭐⭐⭐"
            elif r2 > 0.6:
                performance = "Good ⭐⭐"
            elif r2 > 0.4:
                performance = "Fair ⭐"
            else:
                performance = "Needs Improvement"
            
            print(f"{var.replace('_', ' ').title():15}: {performance} (R²={r2:.3f}, MAE={mae:.3f})")

def main():
    """Run advanced weather model testing"""
    tester = AdvancedWeatherTester()
    tester.run_comprehensive_analysis()

if __name__ == "__main__":
    main()