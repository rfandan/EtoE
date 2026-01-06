import os
from mlProject import logger
from mlProject.entity import DataValidationConfig
import pandas as pd
import sweetviz as sv


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config


    
    def validate_all_columns(self)-> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise e
        
    def generate_profiling_report(self):
        """
        Generates an automated EDA report using sweetviz
        """
        try:
            logger.info("Generating automated EDA report...")
            
            # Monkey-patch numpy for sweetviz compatibility with newer numpy versions
            import numpy as np
            if not hasattr(np, 'VisibleDeprecationWarning'):
                np.VisibleDeprecationWarning = type('VisibleDeprecationWarning', (DeprecationWarning,), {})
            
            data = pd.read_csv(self.config.unzip_data_dir)
            report = sv.analyze(data)
            report.show_html(str(self.config.REPORT_FILE), open_browser=False)
            logger.info(f"EDA report generated at: {self.config.REPORT_FILE}")
        except Exception as e:
            logger.error(f"Failed to generate EDA report: {e}")
            raise e
