import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def main():
    print("Loading cocoon dataset...")
    # Load dataset
    df = pd.read_excel('synthetic_karnataka_cocoon_dataset_10000_records.xlsx')
    
    print(f"Loaded {len(df)} records.")
    
    # Preprocess date
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Feature Engineering from Date
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfYear'] = df['Date'].dt.dayofyear
    # Number of days since a start date to learn linear/nonlinear trends over time
    start_date = pd.to_datetime('2021-01-01')
    df['DateOrdinal'] = (df['Date'] - start_date).dt.days
    
    # Label encode categorical columns
    le_market = LabelEncoder()
    df['Market_encoded'] = le_market.fit_transform(df['Market'])
    
    le_variety = LabelEncoder()
    df['Variety_encoded'] = le_variety.fit_transform(df['Variety'])
    
    # Compute metadata and defaults for UI auto-population
    # We will compute average quality features grouped by Variety and a general overall average
    defaults_by_variety = {}
    for var in df['Variety'].unique():
        var_encoded = int(le_variety.transform([var])[0])
        var_df = df[df['Variety'] == var]
        
        # Calculate stats for the features
        stats = {
            'AvgWeight_g': {
                'min': float(var_df['AvgWeight_g'].min()),
                'max': float(var_df['AvgWeight_g'].max()),
                'mean': float(var_df['AvgWeight_g'].mean())
            },
            'ShellWeight_g': {
                'min': float(var_df['ShellWeight_g'].min()),
                'max': float(var_df['ShellWeight_g'].max()),
                'mean': float(var_df['ShellWeight_g'].mean())
            },
            'ShellRatio_pct': {
                'min': float(var_df['ShellRatio_pct'].min()),
                'max': float(var_df['ShellRatio_pct'].max()),
                'mean': float(var_df['ShellRatio_pct'].mean())
            },
            'FilamentLength_m': {
                'min': float(var_df['FilamentLength_m'].min()),
                'max': float(var_df['FilamentLength_m'].max()),
                'mean': float(var_df['FilamentLength_m'].mean())
            },
            'Reelability_pct': {
                'min': float(var_df['Reelability_pct'].min()),
                'max': float(var_df['Reelability_pct'].max()),
                'mean': float(var_df['Reelability_pct'].mean())
            },
            'Moisture_pct': {
                'min': float(var_df['Moisture_pct'].min()),
                'max': float(var_df['Moisture_pct'].max()),
                'mean': float(var_df['Moisture_pct'].mean())
            },
            'Defects_pct': {
                'min': float(var_df['Defects_pct'].min()),
                'max': float(var_df['Defects_pct'].max()),
                'mean': float(var_df['Defects_pct'].mean())
            },
            'Temp_C': {
                'min': float(var_df['Temp_C'].min()),
                'max': float(var_df['Temp_C'].max()),
                'mean': float(var_df['Temp_C'].mean())
            },
            'Humidity_pct': {
                'min': float(var_df['Humidity_pct'].min()),
                'max': float(var_df['Humidity_pct'].max()),
                'mean': float(var_df['Humidity_pct'].mean())
            },
            'Rainfall_mm': {
                'min': float(var_df['Rainfall_mm'].min()),
                'max': float(var_df['Rainfall_mm'].max()),
                'mean': float(var_df['Rainfall_mm'].mean())
            },
            'RawSilkPrice_RsKg': {
                'min': float(var_df['RawSilkPrice_RsKg'].min()),
                'max': float(var_df['RawSilkPrice_RsKg'].max()),
                'mean': float(var_df['RawSilkPrice_RsKg'].mean())
            }
        }
        defaults_by_variety[var] = stats

    # Create grade adjustments (e.g. Premium, Standard, Economy) based on percentiles of the features
    # Premium = 90th percentile of positive features, 10th percentile of defects
    # Standard = 50th percentile
    # Economy = 10th percentile of positive features, 90th percentile of defects
    grade_presets = {
        'Premium': {
            'AvgWeight_g': float(df['AvgWeight_g'].quantile(0.75)),
            'ShellWeight_g': float(df['ShellWeight_g'].quantile(0.85)),
            'ShellRatio_pct': float(df['ShellRatio_pct'].quantile(0.85)),
            'FilamentLength_m': float(df['FilamentLength_m'].quantile(0.85)),
            'Reelability_pct': float(df['Reelability_pct'].quantile(0.85)),
            'Moisture_pct': float(df['Moisture_pct'].quantile(0.30)), # Lower is generally better/drier
            'Defects_pct': float(df['Defects_pct'].quantile(0.10)), # Fewer defects
            'RawSilkPrice_RsKg': float(df['RawSilkPrice_RsKg'].quantile(0.80))
        },
        'Standard': {
            'AvgWeight_g': float(df['AvgWeight_g'].quantile(0.50)),
            'ShellWeight_g': float(df['ShellWeight_g'].quantile(0.50)),
            'ShellRatio_pct': float(df['ShellRatio_pct'].quantile(0.50)),
            'FilamentLength_m': float(df['FilamentLength_m'].quantile(0.50)),
            'Reelability_pct': float(df['Reelability_pct'].quantile(0.50)),
            'Moisture_pct': float(df['Moisture_pct'].quantile(0.50)),
            'Defects_pct': float(df['Defects_pct'].quantile(0.50)),
            'RawSilkPrice_RsKg': float(df['RawSilkPrice_RsKg'].quantile(0.50))
        },
        'Economy': {
            'AvgWeight_g': float(df['AvgWeight_g'].quantile(0.25)),
            'ShellWeight_g': float(df['ShellWeight_g'].quantile(0.15)),
            'ShellRatio_pct': float(df['ShellRatio_pct'].quantile(0.15)),
            'FilamentLength_m': float(df['FilamentLength_m'].quantile(0.15)),
            'Reelability_pct': float(df['Reelability_pct'].quantile(0.15)),
            'Moisture_pct': float(df['Moisture_pct'].quantile(0.70)), # Higher moisture
            'Defects_pct': float(df['Defects_pct'].quantile(0.85)), # More defects
            'RawSilkPrice_RsKg': float(df['RawSilkPrice_RsKg'].quantile(0.20))
        }
    }

    # Features selection
    features = [
        'Market_encoded', 'Variety_encoded', 'Year', 'Month', 'Day', 'DayOfWeek', 
        'DayOfYear', 'DateOrdinal', 'AvgWeight_g', 'ShellWeight_g', 'ShellRatio_pct', 
        'FilamentLength_m', 'Reelability_pct', 'Moisture_pct', 'Defects_pct', 
        'Temp_C', 'Humidity_pct', 'Rainfall_mm', 'RawSilkPrice_RsKg'
    ]
    target = 'CocoonPrice_RsKg'
    
    X = df[features]
    y = df[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Predictions & Evaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print("Model Evaluation:")
    print(f"R² Score:  {r2:.5f}")
    print(f"MAE:       {mae:.3f} Rs/Kg")
    print(f"RMSE:      {rmse:.3f} Rs/Kg")
    
    # Save the model
    model_filename = 'cocoon_price_model.joblib'
    joblib.dump(model, model_filename)
    print(f"Saved model to {model_filename}")
    
    # Save encoders and metadata
    metadata = {
        'features': features,
        'markets': list(le_market.classes_),
        'market_mapping': {m: int(le_market.transform([m])[0]) for m in le_market.classes_},
        'varieties': list(le_variety.classes_),
        'variety_mapping': {v: int(le_variety.transform([v])[0]) for v in le_variety.classes_},
        'defaults_by_variety': defaults_by_variety,
        'grade_presets': grade_presets,
        'metrics': {
            'r2': float(r2),
            'mae': float(mae),
            'rmse': float(rmse)
        }
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    print("Saved model metadata and defaults to model_metadata.json")

if __name__ == '__main__':
    main()
