import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
import keras

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
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
                    ("one_hot_encoder", OneHotEncoder()),
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

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "master_signal"
            numerical_columns = ["Year", "Month", "Day", "Hour", "Minute"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

            logging.info("Column names in input_feature_train_df: %s", input_feature_train_df.columns.tolist())

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            return train_arr, test_arr

        except Exception as e:
            raise CustomException(e, sys)


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Check the class distribution in the training set
            logging.info("Class distribution in training set:")
            logging.info(pd.Series(y_train).value_counts())
            logging.info("Class distribution in test set:")
            logging.info(pd.Series(y_test).value_counts())

            # Early stopping callback
            '''early_stopping = callbacks.EarlyStopping(
                min_delta=0.001,  # minimum amount of change to count as an improvement
                patience=20,      # how many epochs to wait before stopping
                restore_best_weights=True
            )'''

            # Initialize the NN
            model = Sequential()
            model.add(Dense(128, input_dim=X_train.shape[1], activation="relu"))
            model.add(Dense(64, activation="relu"))
            model.add(Dense(32, activation="relu"))
            model.add(Dense(6, activation="softmax"))

            # Compiling the ANN with a custom optimizer
            opt = Adam(learning_rate=0.001)
            model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Train the ANN with validation split and early stopping callback
            history = model.fit(X_train, y_train, batch_size=256, epochs=800, validation_split=0.1)

            # Save the model
            model.save(self.model_trainer_config.trained_model_file_path)

            # Combine train and test data for prediction
            X = np.vstack((X_train, X_test))
            y = np.concatenate((y_train, y_test))

            # Evaluate the model on the combined dataset
            predicted_prob = model.predict(X)
            predicted_class = np.argmax(predicted_prob, axis=1)

            # Log the predicted values and actual values
            logging.info("Predicted probabilities: {}".format(predicted_prob.flatten()))
            logging.info("Predicted classes: {}".format(predicted_class.flatten()))
            logging.info("Actual values: {}".format(y))

            df_predicted_prob = pd.DataFrame(predicted_class.flatten(), columns=['Prediction'])
            df_predicted_prob.to_csv(os.path.join('artifacts', 'predicted_probablity.csv'), index=False)


            # Logging to a file

            r2_square = r2_score(y, predicted_class)
            return model, r2_square

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    train_path = 'path_to_train_csv'
    test_path = 'path_to_test_csv'

    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.initiate_data_transformation(train_path, test_path)

    model_trainer = ModelTrainer()
    model, r2_square = model_trainer.initiate_model_trainer(train_arr, test_arr)

    logging.info(f"Model training completed with R2 Score: {r2_square}")
