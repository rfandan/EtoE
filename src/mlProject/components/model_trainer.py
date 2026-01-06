import pandas as pd
import os
from mlProject import logger
from sklearn.linear_model import ElasticNetCV
import joblib
from mlProject.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column] # ElasticNetCV expects 1D array for y
        test_y = test_data[self.config.target_column]

        # Using ElasticNetCV as requested, with a search space
        # We can also use the specific alpha/l1_ratio from params if we want to "fix" it, 
        # but CV is better for robustness if data changes.
        en_cv = ElasticNetCV(
            l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
            alphas=[0.01, 0.1, 1, 10],
            cv=5,
            random_state=42
        )
        
        en_cv.fit(train_x, train_y)

        logger.info(f"Best alpha: {en_cv.alpha_}")
        logger.info(f"Best l1_ratio: {en_cv.l1_ratio_}")

        joblib.dump(en_cv, os.path.join(self.config.root_dir, self.config.model_name))
