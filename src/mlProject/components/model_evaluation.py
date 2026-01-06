import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlProject.utils.common import save_json
import numpy as np
import joblib
from mlProject.entity import ModelEvaluationConfig
from pathlib import Path
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import dagshub
import wandb
from datetime import datetime

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow_and_wandb(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        # 1. Initialize DagsHub for MLflow
        if self.config.mlflow_uri != "":
            dagshub.init(repo_owner='rfandan', repo_name='EtoE', mlflow=True)
            mlflow.set_registry_uri(self.config.mlflow_uri)
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # 2. Initialize Weights & Biases
        wandb.init(
            project="EtoE-Wine-Quality",
            config=self.config.all_params,
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)
            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)

            # Saving metrics as local
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            # Log to MLflow
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            # Log to W&B
            wandb.log({
                "rmse": rmse,
                "r2": r2,
                "mae": mae
            })

            # Log Feature Importance (Coefficients) to W&B
            if hasattr(model, 'coef_'):
                features = test_x.columns
                coeffs = model.coef_
                data = [[label, val] for (label, val) in zip(features, coeffs)]
                table = wandb.Table(data=data, columns=["feature", "coefficient"])
                wandb.log({"feature_importance": wandb.plot.bar(table, "feature", "coefficient", title="Feature Importance")})

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, name="model", registered_model_name="ElasticNetWineModel")
            else:
                mlflow.sklearn.log_model(model, name="model")
        
        wandb.finish()
