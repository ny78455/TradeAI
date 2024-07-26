import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import load_model
from src.exception import CustomException
from src.logger import logging
from datetime import datetime

# Path configurations
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

new_data_path = os.path.join('artifacts', 'df_cleaned.csv')

class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.h5")
    log_file_path: str = "model_training.log"

class DataTransformation:
    def __init__(self):
        pass

    def get_data_transformer_object(self):
        try:
            numerical_columns = []
            categorical_columns = ['EMASignal', 'isPivot',
       'CHOCH_pattern_detected', 'fibonacci_signal', 'LBD_detected',
       'LBH_detected', 'SR_signal', 'isBreakOut', 'candlestick_signal',
       'Trend', 'signal1', 'buy_signal', 'sell_signal', 'BB_signal',
       'fractal_high', 'fractal_low', 'fractals_high', 'fractals_low',
       'buy_signal1', 'Fractal_signal', 'sell_signal1', 'VSignal',
       'PriceSignal', 'TotSignal', 'SLSignal', 'grid_signal', 'ordersignal',
       'SLSignal_heiken', 'long_signal', 'martiangle_signal',
       'Candle_direction']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def preprocess_new_data(self, preprocessor):
        try:
            new_df = pd.read_csv(new_data_path)

            # Drop the target column if it exists in the new data
            if 'master_signal' in new_df.columns:
                input_feature_new_df = new_df.drop(columns=['master_signal'], axis=1)
            else:
                input_feature_new_df = new_df

            logging.info("Applying preprocessing object on new data")
            logging.info("Column names in input_feature_train_df: %s", input_feature_new_df.columns.tolist())

            preprocessor.fit(input_feature_new_df)
            input_feature_new_arr = preprocessor.transform(input_feature_new_df)
            
            logging.info(f"Shape of transformed data: {input_feature_new_arr.shape}")

            return input_feature_new_arr

        except Exception as e:
            raise CustomException(e, sys)

    def add_dummy_columns(self, data, expected_shape):
        try:
            # Calculate the difference in columns
            current_shape = data.shape[1]
            if current_shape < expected_shape:
                # Add dummy columns to match the expected shape
                diff = expected_shape - current_shape
                dummy_data = np.zeros((data.shape[0], diff))
                data = np.hstack((data, dummy_data))
                logging.info(f"Added {diff} dummy columns to match expected shape")
            return data
        except Exception as e:
            raise CustomException(e, sys)

class ModelPredictor:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def predict_master_signal(self):
        try:
            new_df = pd.read_csv(new_data_path)
            logging.info(f"Loaded new data with shape: {new_df.shape}")

            # Log all column names in new_df
            logging.info("All columns in new_df: %s", new_df.columns.tolist())

            # Initialize data transformer
            data_transformation = DataTransformation()
            preprocessor = data_transformation.get_data_transformer_object()

            # Preprocess new data
            preprocessed_new_data = data_transformation.preprocess_new_data(preprocessor)
            logging.info(f"Shape of preprocessed data for prediction: {preprocessed_new_data.shape}")

            # Load trained model
            model = load_model(self.model_trainer_config.trained_model_file_path)
            logging.info("Loaded trained model")

            # Add dummy columns to match model input shape if needed
            expected_shape = model.input_shape[1]
            preprocessed_new_data = data_transformation.add_dummy_columns(preprocessed_new_data, expected_shape)
            logging.info(f"Shape of data after adding dummy columns: {preprocessed_new_data.shape}")

            # Make predictions
            predictions = model.predict(preprocessed_new_data)
            predicted_class = np.argmax(predictions, axis=1)
            logging.info(f"Predictions shape: {predictions.shape}, Predicted classes shape: {predicted_class.shape}")

            # Log and return predictions
            df_predictions = pd.DataFrame(predicted_class, columns=['Prediction'])
            logging.info(f"Predictions DataFrame shape: {df_predictions.shape}")

            # Add datetime column in UTC format
            datetime_cols = ["Year", "Month", "Day", "Hour", "Minute"]
            df_datetime = new_df[datetime_cols].copy()
            df_datetime["Datetime"] = pd.to_datetime(df_datetime)
            df_datetime['Datetime'] = df_datetime['Datetime'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
            logging.info("Datetime column created")

            data = pd.read_csv(os.path.join('artifacts', 'df_new.csv'))
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            data.set_index('Datetime', inplace=True)
            logging.info("Loaded existing data with datetime index")

            # Concatenate predictions with datetime column
            df_final_predictions = pd.concat([df_datetime["Datetime"], df_predictions], axis=1)
            df_final_predictions.set_index('Datetime', inplace=True)
            logging.info(f"Final predictions DataFrame shape: {df_final_predictions.shape}")

            df_final_predictions = data.merge(df_final_predictions, left_index=True, right_index=True, how='left')
            logging.info("Merged predictions with existing data")
            df_last_5_rows = pd.read_csv('check1.csv', index_col='Datetime')
            combined_df = pd.concat([df_last_5_rows, df_final_predictions.tail(5)])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]

            combined_df['pointpos'] = combined_df['Low'] + 0.5e-3
            combined_df['Prediction_plot'] = np.where(combined_df['Prediction'] // 10 == 2, combined_df['Prediction'] % 10, 1)

            # Save to CSV
            output_path = os.path.join('artifacts', 'predictions_new.csv')
            df_final_predictions.to_csv(output_path, index=True)
            combined_df.to_csv(os.path.join('templates', 'check1.csv'), index=True)
            combined_df.to_csv('check1.csv', index=True)
            logging.info(f"Predictions saved to {output_path}")

            return df_final_predictions

        except Exception as e:
            logging.error(f"Error in predicting master signal: {str(e)}")
            raise CustomException(e, sys)

# Example of usage
if __name__ == "__main__":
    predictor = ModelPredictor()
    predictions = predictor.predict_master_signal()
    logging.info(f"Predicted master signals: \n{predictions}")
