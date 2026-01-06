import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
import os
from datetime import datetime


class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        self.preprocessor = joblib.load(Path('artifacts/data_transformation/preprocessor.joblib'))
        
        # Define the column names as per the schema
        self.cols = [
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ]
        
        # Path for inference logging
        self.log_path = Path("artifacts/predictions/inference_log.csv")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    
    def predict(self, data):
        # Convert input to DataFrame with correct column names
        data_df = pd.DataFrame(data, columns=self.cols)
        
        # Transform the data using the saved preprocessor
        transformed_data = self.preprocessor.transform(data_df)
        
        # Convert transformed data back to DataFrame to maintain feature names
        transformed_df = pd.DataFrame(transformed_data, columns=self.cols)
        
        # Predict using the loaded model
        prediction = self.model.predict(transformed_df)

        # Log the inference
        self._log_inference(data_df, prediction)

        return prediction

    def _log_inference(self, input_df, prediction):
        """Logs the input features and prediction to a CSV file with a timestamp."""
        log_df = input_df.copy()
        log_df['prediction'] = prediction
        log_df['timestamp'] = datetime.now().strftime("%d %B %Y %H:%M:%S:")

        
        # Append to CSV if it exists, otherwise create it
        if not os.path.isfile(self.log_path):
            log_df.to_csv(self.log_path, index=False)
        else:
            log_df.to_csv(self.log_path, mode='a', header=False, index=False)
