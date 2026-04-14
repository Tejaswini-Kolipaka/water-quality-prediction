
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.spatial.distance import cosine

warnings.filterwarnings("ignore")

# Load data from CSV
df = pd.read_csv("/content/processed_1.csv")
# Convert year to numeric type and handle missing values
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year'])  # Remove rows with missing years

def safe_numeric_conversion(series):
    """Safely convert to numeric, replacing invalid values with median"""
    numeric_series = pd.to_numeric(series, errors='coerce')
    median_value = numeric_series.median()
    return numeric_series.fillna(median_value)

# Convert all parameter columns to numeric safely
parameters = ['ph', 'do', 'bod', 'temp', 'total col']
for param in parameters:
    df[param] = safe_numeric_conversion(df[param])

# Enhanced preprocessing for total coliform with safety checks
def preprocess_total_coliform(data):
    # Ensure no negative or zero values before log transform
    data = np.maximum(data, 0.01)  # Set minimum value to 0.01

    # Apply log transformation
    data_log = np.log1p(data)

    # Calculate quartiles safely
    Q1 = np.percentile(data_log, 25)
    Q3 = np.percentile(data_log, 75)
    IQR = Q3 - Q1

    # Handle outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Replace outliers with median
    median_val = np.median(data_log)
    data_clean = np.where((data_log < lower_bound) | (data_log > upper_bound), median_val, data_log)

    return pd.Series(data_clean)

# Preprocess Dissolved Oxygen (DO)
def preprocess_do(data):
    # Handle missing values
    data = data.fillna(data.median())

    # Detect and handle outliers using IQR
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Replace outliers with the median
    median_val = data.median()
    data_clean = np.where((data < lower_bound) | (data > upper_bound), median_val, data)

    return pd.Series(data_clean)

# Enhanced preprocessing for Temperature (Temp)
def preprocess_temp(data):
    """
    Enhanced preprocessing for temperature data.
    Includes outlier detection, smoothing, and improved handling of missing values.
    """
    # Handle missing values using interpolation
    data = data.interpolate(method='linear', limit_direction='both')

    # Apply smoothing using rolling mean to reduce noise
    smoothed_data = data.rolling(window=3, min_periods=1, center=True).mean()

    # Detect and handle outliers using IQR
    Q1 = np.percentile(smoothed_data, 25)
    Q3 = np.percentile(smoothed_data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Replace outliers with the rolling median
    rolling_median = smoothed_data.rolling(window=5, min_periods=1, center=True).median()
    cleaned_data = np.where((smoothed_data < lower_bound) | (smoothed_data > upper_bound), rolling_median, smoothed_data)

    return pd.Series(cleaned_data)

# Improved handling of missing values with monthly and yearly patterns
for param in parameters:
    # Fill missing values using multiple methods
    monthly_median = df.groupby('month')[param].transform('median')
    yearly_median = df.groupby('year')[param].transform('median')
    overall_median = df[param].median()

    df[param] = df[param].fillna(monthly_median)
    df[param] = df[param].fillna(yearly_median)
    df[param] = df[param].fillna(overall_median)

# Special processing for total coliform
df['total col_processed'] = preprocess_total_coliform(df['total col'])

# Apply preprocessing to DO and Temp
df['do_processed'] = preprocess_do(df['do'])
df['temp_processed'] = preprocess_temp(df['temp'])

# Add month number and seasonal features with validation
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
df['Month_Num'] = pd.Categorical(df['month'], categories=months).codes + 1

# Ensure Month_Num is valid before calculating features
df['Month_Num'] = df['Month_Num'].clip(1, 12)

# Calculate seasonal features
df['sin_month'] = np.sin(2 * np.pi * df['Month_Num'] / 12)
df['cos_month'] = np.cos(2 * np.pi * df['Month_Num'] / 12)
df['sin_month2'] = np.sin(4 * np.pi * df['Month_Num'] / 12)
df['cos_month2'] = np.cos(4 * np.pi * df['Month_Num'] / 12)

# Calculate Year_Num safely
df['Year_Num'] = df['year'] - df['year'].min()

def create_features(data, month_nums, sin_vals, cos_vals, years):
    features = pd.DataFrame()
    features['month_num'] = month_nums
    features['sin_month'] = sin_vals
    features['cos_month'] = cos_vals
    features['year_num'] = years
    features['value'] = data

    # Calculate rolling statistics safely
    for i in range(1, 4):
        features[f'prev_value_{i}'] = features['value'].shift(i)

    features['rolling_mean_3'] = features['value'].rolling(window=3, min_periods=1).mean()
    features['rolling_mean_6'] = features['value'].rolling(window=6, min_periods=1).mean()
    features['rolling_std'] = features['value'].rolling(window=3, min_periods=1).std()
    features['yearly_mean'] = features['value'].rolling(window=12, min_periods=1).mean()

    # Fill any remaining NaN values
    return features.fillna(method='bfill').fillna(method='ffill')

def train_lstm_model(X_lstm, y_lstm):
    model = Sequential([
        Bidirectional(LSTM(128, input_shape=(X_lstm.shape[1], 1), return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        LSTM(32),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')

    try:
        model.fit(X_lstm, y_lstm, epochs=400, batch_size=32, verbose=0, callbacks=[early_stop])
    except Exception as e:
        print(f"Error in LSTM training: {e}")
        return None

    return model

def calculate_similarity(y_true, y_pred):
    """Calculate similarity index (1 - cosine distance)"""
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    
    # Handle cases where all values are zero
    if np.all(y_true == 0) and np.all(y_pred == 0):
        return 1.0
    
    # Calculate cosine similarity
    cosine_dist = cosine(y_true, y_pred)
    similarity = 1 - cosine_dist
    return max(0, similarity)  # Ensure similarity is between 0 and 1

def train_random_forest(features_df, target):
    """Train and optimize a Random Forest model"""
    try:
        # Prepare data
        X = features_df.drop(columns=['value'])
        y = target

        # Remove rows where target is NaN
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]

        # Train-test split (time-based)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Basic Random Forest
        rf = RandomForestRegressor(n_estimators=200,
                                 max_depth=10,
                                 min_samples_split=5,
                                 random_state=42,
                                 n_jobs=-1)

        # Train model
        rf.fit(X_train, y_train)

        # Evaluate
        y_pred = rf.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        sim = calculate_similarity(y_test, y_pred)

        return rf, {'rmse': rmse, 'mae': mae, 'mape': mape, 'sim': sim}
    except Exception as e:
        print(f"Error in Random Forest training: {e}")
        return None, None

def safe_arima_prediction(model, steps=1):
    """Safely make ARIMA predictions with fallback"""
    try:
        pred = float(model.forecast(steps=steps)[0])
        if np.isnan(pred):
            return None
        return pred
    except Exception:
        return None

def train_ensemble_model(param_data, month_nums, sin_vals, cos_vals, years):
    try:
        features_df = create_features(param_data, month_nums, sin_vals, cos_vals, years)

        # Choose scaler based on parameter
        scaler = RobustScaler() if 'total col' in str(param_data.name) else MinMaxScaler(feature_range=(0, 1))

        # Scale data safely
        data_for_scaling = param_data.values.reshape(-1, 1)
        data_for_scaling = np.nan_to_num(data_for_scaling, nan=np.nanmedian(data_for_scaling))
        scaled_data = scaler.fit_transform(data_for_scaling)

        # Prepare sequences
        seq_length = 6
        X_lstm, y_lstm = [], []
        for i in range(len(scaled_data) - seq_length):
            X_lstm.append(scaled_data[i:(i + seq_length)])
            y_lstm.append(scaled_data[i + seq_length])

        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)

        # Train models with error handling
        lstm_model = train_lstm_model(X_lstm, y_lstm)

        # Train Random Forest
        rf_model, metrics = train_random_forest(features_df, param_data)

        try:
            arima_model = ARIMA(param_data.values, order=(2,1,2)).fit()
        except Exception:
            arima_model = None

        return {
            'lstm': lstm_model,
            'arima': arima_model,
            'random_forest': rf_model,
            'scaler': scaler,
            'last_sequence': scaled_data[-seq_length:],
            'features': features_df,
            'metrics': metrics
        }
    except Exception as e:
        print(f"Error in ensemble model training: {e}")
        return None

def predict_parameter(param, models):
    try:
        model_dict = models[param]
        if model_dict is None:
            return None

        predictions = []
        weights = []

        # LSTM prediction
        if model_dict['lstm'] is not None:
            lstm_input = model_dict['last_sequence'].reshape(1, 6, 1)
            lstm_pred = model_dict['lstm'].predict(lstm_input, verbose=0)[0][0]

            if param == 'total col':
                lstm_pred = model_dict['scaler'].inverse_transform([[lstm_pred]])[0][0]
                lstm_pred = np.expm1(lstm_pred)
            else:
                lstm_pred = model_dict['scaler'].inverse_transform([[lstm_pred]])[0][0]

            predictions.append(lstm_pred)
            weights.append(0.4)  # Weight for LSTM

        # ARIMA prediction
        if model_dict['arima'] is not None:
            arima_pred = safe_arima_prediction(model_dict['arima'])
            if arima_pred is not None:
                predictions.append(arima_pred)
                weights.append(0.3)  # Weight for ARIMA

        # Random Forest prediction
        if model_dict['random_forest'] is not None:
            try:
                # Prepare features for prediction
                last_features = model_dict['features'].iloc[-1:].drop(columns=['value'])
                rf_pred = model_dict['random_forest'].predict(last_features)[0]

                predictions.append(rf_pred)
                weights.append(0.3)  # Weight for Random Forest
            except Exception as e:
                print(f"Random Forest prediction error: {e}")

        # Calculate weighted average
        if len(predictions) > 0:
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w/total_weight for w in weights]

            # Calculate weighted prediction
            prediction = sum(p * w for p, w in zip(predictions, normalized_weights))
        else:
            # Fallback to historical median if all models fail
            prediction = df[param].median()

        # Final validation
        if param == 'total col':
            prediction = max(0, prediction)  # Ensure non-negative
        elif param == 'ph':
            prediction = np.clip(prediction, 0, 14)  # Valid pH range
        else:
            prediction = max(0, prediction)  # Non-negative for other parameters

        return prediction

    except Exception as e:
        print(f"Error in prediction for {param}: {e}")
        return df[param].median()  # Fallback to median

# Train models
models = {}
all_metrics = {}
parameters = {'ph': 'ph', 'do': 'do_processed', 'bod': 'bod', 'temp': 'temp_processed', 'total col': 'total col_processed'}

for param_key, param_col in parameters.items():
    model_dict = train_ensemble_model(
        df[param_col],
        df['Month_Num'],
        df['sin_month'],
        df['cos_month'],
        df['Year_Num']
    )
    models[param_key] = model_dict

# Prediction function with metrics display
def get_predictions(year_to_predict):
    if year_to_predict <= df['year'].max():
        return f"Error: The dataset contains data up to {df['year'].max()}. Please enter a year after {df['year'].max()}."

    predictions = {}
    metrics_output = "\n📊 Model Evaluation Metrics by Parameter:\n"
    
    # Print metrics for each parameter
    for param in parameters.keys():
        model_dict = models[param]
        if model_dict and 'metrics' in model_dict and model_dict['metrics']:
            metrics = model_dict['metrics']
            metrics_output += f"\n{param.upper()}:\n"
            metrics_output += f"  RMSE: {metrics['rmse']:.4f}\n"
            metrics_output += f"  MAE: {metrics['mae']:.4f}\n"
            metrics_output += f"  MAPE: {metrics['mape']:.4f}\n"
            metrics_output += f"  Similarity Index: {metrics['sim']:.4f}\n"
    
    # Get predictions
    for param in parameters.keys():
        pred_value = predict_parameter(param, models)
        if pred_value is not None:
            predictions[param] = pred_value
        else:
            predictions[param] = df[param].median()  # Fallback to median

    return metrics_output + \
           f"\n✅ Predicted Values for {year_to_predict}:\n" + \
           f"pH: {predictions['ph']:.2f}\n" + \
           f"Dissolved Oxygen (DO) (mg/L): {predictions['do']:.2f}\n" + \
           f"Biochemical Oxygen Demand (BOD) (mg/L): {predictions['bod']:.2f}\n" + \
           f"Temperature (° C): {predictions['temp']:.2f}\n" + \
           f"Total Coliform (MPN/100mL): {predictions['total col']:.2f}"

# Get user input and display results
year_to_predict = int(input("Enter the year you want to predict (e.g., 2024): ").strip())
print(get_predictions(year_to_predict))
