import os
from mlProject import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
import pandas as pd
import numpy as np
from mlProject.entity import DataTransformationConfig
from mlProject.utils.common import save_bin
from pathlib import Path

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)

        # Split the data into training and test sets. (0.8, 0.2) split as per research.
        train, test = train_test_split(data, test_size=0.2, random_state=42)

        # Separate X and y
        X_train = train.drop(["quality"], axis=1)
        y_train = train["quality"]
        X_test = test.drop(["quality"], axis=1)
        y_test = test["quality"]

        # Initialize transformers
        pt = PowerTransformer(method='yeo-johnson')
        scaler = StandardScaler()

        # Fit and transform training data
        X_train_transformed = pt.fit_transform(X_train)
        X_train_transformed = scaler.fit_transform(X_train_transformed)

        # Transform test data
        X_test_transformed = pt.transform(X_test)
        X_test_transformed = scaler.transform(X_test_transformed)

        # Convert back to DataFrame to keep structure (optional but good for CSV)
        X_train_final = pd.DataFrame(X_train_transformed, columns=X_train.columns)
        X_test_final = pd.DataFrame(X_test_transformed, columns=X_test.columns)

        # Add target back
        train_final = pd.concat([X_train_final, y_train.reset_index(drop=True)], axis=1)
        test_final = pd.concat([X_test_final, y_test.reset_index(drop=True)], axis=1)

        # Save transformed data
        train_final.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test_final.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        # Save the preprocessor (PowerTransformer + Scaler)
        # We can save them as a list or a pipeline
        from sklearn.pipeline import Pipeline
        preprocessor = Pipeline([
            ('transform', pt),
            ('scaler', scaler)
        ])
        save_bin(data=preprocessor, path=Path(os.path.join(self.config.root_dir, "preprocessor.joblib")))

        logger.info("Transformed data and saved to artifacts")
        logger.info(f"Train shape: {train_final.shape}")
        logger.info(f"Test shape: {test_final.shape}")

        print(train_final.shape)
        print(test_final.shape)
