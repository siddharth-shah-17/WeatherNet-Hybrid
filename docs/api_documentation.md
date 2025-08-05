# API Documentation

## 🚀 Model Interface

# API Documentation

## 🌐 API Overview

The Weather Prediction Model provides a comprehensive API for training, evaluation, and inference. This documentation covers all public interfaces, configuration options, and usage examples for integrating the model into production systems.

### API Design Principles

1. **Intuitive Interface**: Simple and consistent method signatures
2. **Comprehensive Configuration**: Extensive customization options
3. **Error Handling**: Robust error reporting and validation
4. **Performance**: Optimized for production deployment
5. **Extensibility**: Modular design for easy enhancement

## 🏗️ Core Classes and Methods

### WeatherPredictor Class

The main interface for model training and prediction.

```python
class WeatherPredictor:
    """
    Advanced Weather Prediction Model
    
    A hybrid neural network combining CNN, LSTM, and Attention mechanisms
    for multi-horizon weather forecasting.
    """
    
    def __init__(self, sequence_length: int = 168, prediction_horizon: int = 24):
        """
        Initialize Weather Prediction Model.
        
        Args:
            sequence_length (int): Number of hours to look back (default: 168 = 7 days)
            prediction_horizon (int): Number of hours to predict ahead (default: 24 = 1 day)
        
        Attributes:
            sequence_length (int): Input sequence length
            prediction_horizon (int): Output prediction length
            scalers (dict): Feature normalization scalers
            model (tf.keras.Model): Trained neural network model
            history (tf.keras.callbacks.History): Training history
        """
```

#### Training Methods

##### `create_sample_data(n_samples, save_to_file)`
```python
def create_sample_data(self, n_samples: int = 10000, save_to_file: bool = True) -> pd.DataFrame:
    """
    Generate synthetic weather data for training and testing.
    
    Creates realistic weather patterns incorporating:
    - Seasonal temperature variations
    - Diurnal cycles
    - Meteorological correlations
    - Geographic context
    
    Args:
        n_samples (int): Number of hourly samples to generate
        save_to_file (bool): Whether to save data to CSV file
    
    Returns:
        pd.DataFrame: Generated weather data with all required features
    
    Example:
        >>> predictor = WeatherPredictor()
        >>> data = predictor.create_sample_data(n_samples=8760)  # 1 year
        >>> print(data.shape)
        (8760, 16)
    """
```

##### `train_model(df, csv_file, validation_split, epochs, batch_size)`
```python
def train_model(self, 
                df: Optional[pd.DataFrame] = None,
                csv_file: Optional[str] = None,
                validation_split: float = 0.2,
                epochs: int = 100,
                batch_size: int = 32) -> tf.keras.callbacks.History:
    """
    Train the weather prediction model.
    
    Args:
        df (pd.DataFrame, optional): Training data as DataFrame
        csv_file (str, optional): Path to CSV file with training data
        validation_split (float): Fraction of data for validation (0.0-1.0)
        epochs (int): Maximum number of training epochs
        batch_size (int): Training batch size
    
    Returns:
        tf.keras.callbacks.History: Training history with metrics
    
    Raises:
        ValueError: If neither df nor csv_file is provided
        FileNotFoundError: If csv_file doesn't exist
        ValidationError: If data format is invalid
    
    Example:
        >>> predictor = WeatherPredictor()
        >>> history = predictor.train_model(
        ...     csv_file='weather_data.csv',
        ...     epochs=50,
        ...     batch_size=64
        ... )
        >>> print(f"Final loss: {history.history['loss'][-1]:.4f}")
    """
```

#### Prediction Methods

##### `predict_weather(input_sequence)`
```python
def predict_weather(self, input_sequence: np.ndarray) -> np.ndarray:
    """
    Generate weather predictions for the next prediction_horizon hours.
    
    Args:
        input_sequence (np.ndarray): Input weather data
            Shape: (sequence_length, n_features) or (1, sequence_length, n_features)
    
    Returns:
        np.ndarray: Weather predictions
            Shape: (prediction_horizon, n_target_variables)
    
    Raises:
        ValueError: If model not trained or input shape invalid
        
    Example:
        >>> # Assuming you have 168 hours of recent weather data
        >>> recent_data = load_recent_weather_data()  # Shape: (168, 19)
        >>> predictions = predictor.predict_weather(recent_data)
        >>> print(f"Next 24h temperature: {predictions[:, 0]}")
    """
```

#### Utility Methods

##### `save_model(filepath)` / `load_model(filepath)`
```python
def save_model(self, filepath: str = 'weather_predictor_model.keras') -> None:
    """
    Save trained model to disk.
    
    Args:
        filepath (str): Path where to save the model
    
    Example:
        >>> predictor.save_model('models/production_model.keras')
    """

def load_model(self, filepath: str = 'weather_predictor_model.keras') -> None:
    """
    Load trained model from disk.
    
    Args:
        filepath (str): Path to saved model file
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        
    Example:
        >>> predictor = WeatherPredictor()
        >>> predictor.load_model('models/production_model.keras')
    """
```

### WeatherInference Class

Production-ready inference interface with additional features.

```python
class WeatherInference:
    """
    Production-ready weather prediction inference system.
    
    Provides enhanced prediction capabilities including:
    - Batch predictions
    - Confidence estimation
    - JSON output formatting
    - Error handling
    """
    
    def __init__(self, model_path: str = 'models/advanced_weather_model.keras'):
        """
        Initialize inference system.
        
        Args:
            model_path (str): Path to trained model file
        """
```

#### Core Inference Methods

##### `predict_single_sequence(input_data, return_confidence)`
```python
def predict_single_sequence(self, 
                          input_data: Union[pd.DataFrame, np.ndarray],
                          return_confidence: bool = False) -> Dict:
    """
    Make weather predictions for a single input sequence.
    
    Args:
        input_data: Input weather data (last 168 hours)
        return_confidence: Whether to include confidence metrics
    
    Returns:
        Dict: Structured prediction results
        
    Example:
        >>> inference = WeatherInference()
        >>> inference.load_model()
        >>> results = inference.predict_single_sequence(recent_data)
        >>> temp_forecast = results['predictions']['temperature']['values']
    """
```

##### `predict_from_csv(csv_file, output_file)`
```python
def predict_from_csv(self, 
                    csv_file: str, 
                    output_file: Optional[str] = None) -> Dict:
    """
    Make predictions from CSV file input.
    
    Args:
        csv_file: Path to input CSV file
        output_file: Optional path to save results
    
    Returns:
        Dict: Prediction results in structured format
    """
```

### AdvancedWeatherTester Class

Comprehensive model evaluation and testing.

```python
class AdvancedWeatherTester:
    """
    Advanced evaluation and testing framework for weather models.
    
    Provides comprehensive analysis including:
    - Multi-metric evaluation
    - Seasonal performance analysis
    - Forecast horizon analysis
    - Visualization generation
    """
    
    def __init__(self, model_path: str = 'advanced_weather_model.keras'):
        """
        Initialize the testing framework.
        
        Args:
            model_path (str): Path to model file to evaluate
        """
```

#### Evaluation Methods

##### `evaluate_model_accuracy(csv_file)`
```python
def evaluate_model_accuracy(self, csv_file: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Comprehensive model accuracy evaluation.
    
    Args:
        csv_file (str): Path to evaluation dataset
    
    Returns:
        Tuple containing:
        - predictions (np.ndarray): Model predictions
        - actuals (np.ndarray): Ground truth values  
        - metrics (Dict): Evaluation metrics by variable
    
    Example:
        >>> tester = AdvancedWeatherTester()
        >>> pred, actual, metrics = tester.evaluate_model_accuracy('test_data.csv')
        >>> print(f"Temperature R²: {metrics['temperature']['R²']:.3f}")
    """
```

## 📝 Configuration Options

### Model Architecture Configuration
```python
architecture_config = {
    'cnn_filters': [64, 128],         # CNN filter sizes
    'lstm_units': [128, 64],          # LSTM layer sizes
    'attention_heads': 8,             # Multi-head attention
    'dense_units': [256, 128, 64],    # Dense layer sizes
    'dropout_rates': [0.3, 0.2, 0.1], # Dropout rates
}
```

### Training Configuration
```python
training_config = {
    'optimizer': 'adam',              # Optimizer type
    'learning_rate': 0.001,           # Initial learning rate
    'loss_function': 'huber',         # Loss function
    'metrics': ['mae', 'mse'],        # Training metrics
    'early_stopping_patience': 15,   # Early stopping patience
    'reduce_lr_patience': 7,          # LR reduction patience
    'reduce_lr_factor': 0.5,          # LR reduction factor
}
```

### Data Configuration
```python
data_config = {
    'sequence_length': 168,           # Input sequence length
    'prediction_horizon': 24,         # Output sequence length
    'target_variables': [             # Variables to predict
        'temperature', 
        'humidity', 
        'precipitation', 
        'wind_speed'
    ],
    'feature_columns': [              # Input features
        'temperature', 'humidity', 'pressure', 'wind_speed', 
        'precipitation', 'cloud_cover', 'uv_index', 'visibility',
        'latitude', 'longitude', 'elevation',
        'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
        'day_sin', 'day_cos', 'wind_dir_sin', 'wind_dir_cos'
    ]
}
```

## 🔧 Usage Examples

### Basic Training Example
```python
from src.training import WeatherPredictor

# Initialize model
predictor = WeatherPredictor(sequence_length=168, prediction_horizon=24)

# Train with your data
history = predictor.train_model(
    csv_file='data/weather_data.csv',
    validation_split=0.2,
    epochs=100,
    batch_size=32
)

# Save trained model
predictor.save_model('models/my_weather_model.keras')

# Plot training results
predictor.plot_training_history()
```

### Production Inference Example
```python
from src.inference import WeatherInference

# Initialize inference system
inference = WeatherInference('models/production_model.keras')

# Load model
if inference.load_model():
    # Make prediction from CSV
    results = inference.predict_from_csv(
        csv_file='data/recent_weather.csv',
        output_file='results/forecast.json'
    )
    
    # Generate human-readable summary
    summary = inference.create_forecast_summary(results)
    print(summary)
```

### Model Evaluation Example
```python
from src.model_evaluation import AdvancedWeatherTester

# Initialize tester
tester = AdvancedWeatherTester('models/model_to_evaluate.keras')

# Run comprehensive analysis
tester.run_comprehensive_analysis()

# Get specific metrics
predictions, actuals, metrics = tester.evaluate_model_accuracy('test_data.csv')

# Check performance by variable
for variable in ['temperature', 'humidity', 'precipitation', 'wind_speed']:
    r2 = metrics[variable]['R²']
    mae = metrics[variable]['MAE']
    print(f"{variable}: R²={r2:.3f}, MAE={mae:.3f}")
```

### Custom Data Pipeline Example
```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess custom data
def preprocess_custom_data(df):
    # Add cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Add other required features...
    
    # Scale features
    scaler = MinMaxScaler()
    feature_columns = [...]  # Define your features
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    return df, scaler

# Use with model
df = pd.read_csv('my_weather_data.csv')
df_processed, scaler = preprocess_custom_data(df)

predictor = WeatherPredictor()
history = predictor.train_model(df=df_processed)
```

## 🚨 Error Handling

### Common Exceptions

#### `ValueError`
- Invalid input data shape
- Missing required model
- Invalid configuration parameters

#### `FileNotFoundError`
- Model file doesn't exist
- Data file not found

#### `ValidationError`
- Data format validation failures
- Missing required columns

### Error Handling Best Practices
```python
try:
    predictor = WeatherPredictor()
    history = predictor.train_model(csv_file='data.csv')
except FileNotFoundError:
    print("Data file not found. Please check the file path.")
except ValueError as e:
    print(f"Invalid configuration: {e}")
except Exception as e:
    print(f"Unexpected error during training: {e}")
```

## 📊 Output Formats

### Prediction Output Format
```json
{
  "predictions": {
    "temperature": {
      "values": [20.5, 21.2, 22.1, ...],
      "hours_ahead": [1, 2, 3, ..., 24],
      "unit": "°C",
      "confidence": 0.85
    },
    "humidity": {
      "values": [65.2, 66.1, 67.5, ...],
      "hours_ahead": [1, 2, 3, ..., 24],
      "unit": "%",
      "confidence": 0.78
    }
  },
  "metadata": {
    "model_path": "models/production_model.keras",
    "prediction_horizon": 24,
    "timestamp": "2024-01-15T10:30:00",
    "input_sequence_length": 168
  }
}
```

### Evaluation Metrics Output
```python
{
    'temperature': {
        'MAE': 2.45,
        'RMSE': 3.12,
        'R²': 0.87,
        'MAPE': 12.5
    },
    'humidity': {
        'MAE': 8.20,
        'RMSE': 11.45,
        'R²': 0.72,
        'MAPE': 15.8
    }
    # ... other variables
}
```

## 🔗 Integration Guide

### REST API Wrapper Example
```python
from flask import Flask, request, jsonify
from src.inference import WeatherInference

app = Flask(__name__)
inference = WeatherInference()
inference.load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Convert to appropriate format
        results = inference.predict_single_sequence(data['weather_data'])
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Docker Deployment Example
```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

CMD ["python", "src/inference.py"]
```

This comprehensive API documentation provides all necessary information for integrating and using the weather prediction model in various production scenarios.

## 📋 Core Classes

### WeatherPredictor Class

```python
from src.training import WeatherPredictor

class WeatherPredictor:
    def __init__(self, sequence_length=168, prediction_horizon=24):
        """
        Initialize Weather Prediction Model
        
        Args:
            sequence_length (int): Input sequence length in hours (default: 168 = 7 days)
            prediction_horizon (int): Prediction horizon in hours (default: 24 = 1 day)
        """
```

#### Methods

##### Training Methods
```python
def create_sample_data(self, n_samples=10000, save_to_file=True):
    """
    Generate synthetic weather data for testing
    
    Args:
        n_samples (int): Number of hourly samples to generate
        save_to_file (bool): Whether to save data to CSV file
        
    Returns:
        pandas.DataFrame: Generated weather data
    """

def preprocess_data(self, df):
    """
    Preprocess weather data for model training
    
    Args:
        df (pandas.DataFrame): Raw weather data
        
    Returns:
        tuple: (features, targets, feature_columns, target_columns)
    """

def train_model(self, df=None, csv_file=None, validation_split=0.2, epochs=100, batch_size=32):
    """
    Train the weather prediction model
    
    Args:
        df (pandas.DataFrame, optional): Weather data DataFrame
        csv_file (str, optional): Path to weather data CSV file
        validation_split (float): Fraction of data for validation
        epochs (int): Maximum training epochs
        batch_size (int): Training batch size
        
    Returns:
        tensorflow.keras.callbacks.History: Training history
    """
```

##### Prediction Methods
```python
def predict_weather(self, input_sequence):
    """
    Make weather predictions for next 24 hours
    
    Args:
        input_sequence (numpy.ndarray): Input weather sequence (168, 19)
        
    Returns:
        numpy.ndarray: Weather predictions (24, 4)
    """

def load_model(self, filepath='weather_predictor_model.keras'):
    """
    Load pre-trained model from file
    
    Args:
        filepath (str): Path to model file
        
    Returns:
        bool: Success status
    """

def save_model(self, filepath='weather_predictor_model.keras'):
    """
    Save trained model to file
    
    Args:
        filepath (str): Output file path
    """
```

### AdvancedWeatherTester Class

```python
from src.model_evaluation import AdvancedWeatherTester

class AdvancedWeatherTester:
    def __init__(self, model_path='advanced_weather_model.keras'):
        """
        Initialize model evaluation framework
        
        Args:
            model_path (str): Path to trained model file
        """
```

#### Evaluation Methods
```python
def evaluate_model_accuracy(self, csv_file='data/weather_data_nyc.csv'):
    """
    Comprehensive model accuracy evaluation
    
    Args:
        csv_file (str): Path to test data CSV file
        
    Returns:
        tuple: (predictions, actuals, metrics)
    """

def test_seasonal_performance(self, csv_file='data/weather_data_nyc.csv'):
    """
    Analyze model performance across seasons
    
    Args:
        csv_file (str): Path to weather data CSV file
        
    Returns:
        dict: Seasonal performance metrics
    """

def create_prediction_confidence_analysis(self, csv_file='data/weather_data_nyc.csv'):
    """
    Analyze prediction confidence over forecast horizon
    
    Args:
        csv_file (str): Path to weather data CSV file
        
    Returns:
        dict: Hourly confidence metrics
    """
```

## 🔧 Usage Examples

### Basic Training Example
```python
# Initialize predictor
predictor = WeatherPredictor(sequence_length=168, prediction_horizon=24)

# Train model with CSV data
history = predictor.train_model(
    csv_file='data/weather_data_nyc.csv',
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# Save trained model
predictor.save_model('models/my_weather_model.keras')
```

### Model Evaluation Example
```python
# Initialize tester
tester = AdvancedWeatherTester(model_path='models/advanced_weather_model.keras')

# Run comprehensive evaluation
predictions, actuals, metrics = tester.evaluate_model_accuracy()

# Analyze seasonal performance
seasonal_results = tester.test_seasonal_performance()

# Create confidence analysis
tester.create_prediction_confidence_analysis()
```

### Inference Example
```python
import numpy as np
import pandas as pd

# Load model
predictor = WeatherPredictor()
predictor.load_model('models/advanced_weather_model.keras')

# Prepare input data (last 7 days of weather)
recent_data = pd.read_csv('data/weather_data_nyc.csv').tail(168)
features, _, _, _ = predictor.preprocess_data(recent_data)
input_sequence = features.values

# Make prediction
forecast = predictor.predict_weather(input_sequence)

# Forecast contains 24 hours of predictions for:
# [temperature, humidity, precipitation, wind_speed]
print(f"24-hour forecast shape: {forecast.shape}")  # (24, 4)
```

## 📊 Data Formats

### Input Data Format
```python
# Required CSV columns for training/evaluation
required_columns = [
    'datetime',       # ISO format timestamp
    'temperature',    # °C
    'humidity',       # %
    'pressure',       # hPa
    'wind_speed',     # m/s
    'wind_direction', # degrees (0-360)
    'precipitation',  # mm/h
    'cloud_cover',    # %
    'uv_index',       # index (0-12)
    'visibility',     # km
    'latitude',       # decimal degrees
    'longitude',      # decimal degrees
    'elevation'       # meters
]
```

### Model Input/Output Shapes
```python
# Model expects input shape: (batch_size, 168, 19)
# - 168 hours (7 days) of historical data
# - 19 features (13 weather + 6 derived temporal features)

# Model produces output shape: (batch_size, 24, 4)
# - 24 hours of predictions
# - 4 weather variables: [temperature, humidity, precipitation, wind_speed]
```

## ⚠️ Error Handling

### Common Exceptions
```python
# Model loading errors
try:
    predictor.load_model('nonexistent_model.keras')
except FileNotFoundError:
    print("Model file not found")

# Data validation errors
try:
    features, targets = predictor.preprocess_data(invalid_df)
except KeyError as e:
    print(f"Missing required column: {e}")

# Prediction errors
try:
    forecast = predictor.predict_weather(wrong_shape_input)
except ValueError as e:
    print(f"Input shape error: {e}")
```

### Data Validation
```python
def validate_input_data(df):
    """
    Validate input data format and ranges
    
    Args:
        df (pandas.DataFrame): Input weather data
        
    Raises:
        ValueError: If data validation fails
    """
    # Check required columns
    required_cols = ['datetime', 'temperature', 'humidity', ...]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Check data ranges
    if df['temperature'].min() < -50 or df['temperature'].max() > 60:
        raise ValueError("Temperature values out of realistic range")
    
    # Check for missing values
    if df.isnull().any().any():
        raise ValueError("Data contains missing values")
```

## 🔌 Integration Examples

### Flask API Integration
```python
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)
predictor = WeatherPredictor()
predictor.load_model('models/production_model.keras')

@app.route('/predict', methods=['POST'])
def predict_weather():
    try:
        # Get input data
        input_data = request.json['weather_data']  # List of 168 hourly records
        
        # Convert to numpy array and reshape
        input_array = np.array(input_data).reshape(1, 168, 19)
        
        # Make prediction
        forecast = predictor.predict_weather(input_array)
        
        # Format response
        return jsonify({
            'forecast': forecast.tolist(),
            'horizon_hours': 24,
            'variables': ['temperature', 'humidity', 'precipitation', 'wind_speed']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400
```

### Batch Processing Example
```python
def process_weather_stations(station_data_dir):
    """
    Process multiple weather stations in batch
    
    Args:
        station_data_dir (str): Directory containing station CSV files
    """
    predictor = WeatherPredictor()
    predictor.load_model('models/advanced_weather_model.keras')
    
    results = {}
    
    for csv_file in glob.glob(f"{station_data_dir}/*.csv"):
        station_id = os.path.basename(csv_file).split('.')[0]
        
        # Load station data
        df = pd.read_csv(csv_file)
        
        # Get latest 168 hours
        recent_data = df.tail(168)
        features, _, _, _ = predictor.preprocess_data(recent_data)
        
        # Make prediction
        forecast = predictor.predict_weather(features.values)
        
        results[station_id] = {
            'forecast': forecast.tolist(),
            'timestamp': recent_data['datetime'].iloc[-1]
        }
    
    return results
```

## 🚀 Advanced Usage

### Custom Model Architecture
```python
# Modify model architecture before training
predictor = WeatherPredictor()

# Access model building method
custom_model = predictor.build_advanced_model(
    input_shape=(168, 19),
    output_shape=(24, 4)
)

# Customize architecture
custom_model.summary()

# Assign to predictor
predictor.model = custom_model
```

### Transfer Learning
```python
# Load pre-trained model for transfer learning
base_predictor = WeatherPredictor()
base_predictor.load_model('models/pretrained_global_model.keras')

# Freeze some layers
for layer in base_predictor.model.layers[:5]:
    layer.trainable = False

# Fine-tune on local data
base_predictor.train_model(
    csv_file='data/local_weather_data.csv',
    epochs=20,
    learning_rate=0.0001
)
```

## 📝 Configuration Options

### Training Configuration
```python
training_config = {
    'sequence_length': 168,      # Input sequence length
    'prediction_horizon': 24,    # Forecast horizon
    'batch_size': 32,           # Training batch size
    'epochs': 100,              # Maximum epochs
    'validation_split': 0.2,    # Validation fraction
    'learning_rate': 0.001,     # Initial learning rate
    'patience': 15,             # Early stopping patience
    'min_lr': 1e-6,            # Minimum learning rate
}
```

### Model Architecture Configuration
```python
architecture_config = {
    'cnn_filters': [64, 128],   # CNN filter sizes
    'lstm_units': [128, 64],    # LSTM layer sizes
    'attention_heads': 8,        # Multi-head attention
    'dense_units': [256, 128, 64],  # Dense layer sizes
    'dropout_rates': [0.3, 0.2, 0.1],  # Dropout rates
}
```

This API documentation provides a comprehensive guide for using the weather prediction model in various scenarios, from basic training to advanced production deployments.
