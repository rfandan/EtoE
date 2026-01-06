import random
import pandas as pd
from locust import HttpUser, task, between, events

@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--drift", action="store_true", default=False, help="Enable simulated data drift")

class WineQualityUser(HttpUser):
    # Wait between 1 and 3 seconds between tasks to simulate human-ish or bot behavior
    wait_time = between(1, 3)
    
    def on_start(self):
        """
        Executed when a virtual user starts. 
        We load the test data to pick random samples from it.
        """
        try:
            self.test_data = pd.read_csv("artifacts/data_transformation/test.csv")
            # The CSV has 'quality' column which we should not send to the predict API
            self.features = self.test_data.drop('quality', axis=1)
        except Exception as e:
            print(f"Error loading test data: {e}")
            self.test_data = None

    @task(5)
    def predict_valid_data(self):
        """
        Sends a valid prediction request using a random row from the test set.
        """
        if self.features is not None:
            # Pick a random row
            random_row = self.features.sample(n=1).to_dict(orient='records')[0]
            
            # Send POST request to /predict
            # We use the 'name' parameter so Locust groups these requests in the UI
            self.client.post("/predict", json=random_row, name="/predict [Valid]")

    @task(1)
    def predict_invalid_data(self):
        """
        Sends invalid data to test Pydantic validation.
        """
        invalid_payload = {
            "fixed acidity": "not_a_number", # Should trigger Pydantic error
            "volatile acidity": 0.7,
            "citric acid": 0.0,
            "residual sugar": 1.9,
            "chlorides": 0.076,
            "free sulfur dioxide": 11.0,
            "total sulfur dioxide": 34.0,
            "density": 0.9978,
            "pH": 3.51,
            "sulphates": 0.56,
            "alcohol": 9.4
        }
        
        # We expect a 422 error here, so we tell Locust not to mark it as a failure automatically
        with self.client.post("/predict", json=invalid_payload, name="/predict [Invalid]", catch_response=True) as response:
            if response.status_code == 422:
                response.success()
            else:
                response.failure(f"Expected 422 but got {response.status_code}")

    @task(3)
    def predict_drifted_data(self):
        """
        Sends data that is intentionally 'drifted' (extreme values)
        to trigger Evidently AI drift detection.
        Only runs if --drift is enabled in the UI.
        """
        if self.features is not None and self.environment.parsed_options.drift:
            random_row = self.features.sample(n=1).to_dict(orient='records')[0]
            # Intentionally set alcohol to an extreme value (e.g., 50%)
            random_row['alcohol'] = 50.0 
            self.client.post("/predict", json=random_row, name="Task: Send Drifted Data")
        else:
            # If drift is disabled, just do a normal prediction to keep traffic consistent
            self.predict_valid_data()

    @task(1)
    def trigger_drift_check(self):
        """
        Periodically triggers the background drift calculation in the API.
        """
        self.client.get("/check_drift", name="Trigger Drift Check")

    @task(1)
    def visit_homepage(self):

        """
        Simulates a user visiting the web interface.
        """
        self.client.get("/", name="Homepage")
