# prediction_writer.py

import pandas as pd
from sqlalchemy import create_engine
import os
import uuid

class PredictionWriter:
    def __init__(self, pg_url: str, table="ml_predictions"):
        self.engine = create_engine(pg_url)
        self.table = table

    def write(self, tick: dict, label_15: str, prob_15: float, label_30: str, prob_30: float, quality:float):
        signal_id = str(uuid.uuid4())
        row = {
            "signal_id": signal_id,
            "ts": tick["ts"],
            "prediction": label_15,
            "confidence": round(prob_15, 4),
            "prediction_30": label_30,
            "confidence_30": round(prob_30, 4),
            "quality": round(quality, 4),
            "ltp": tick["ltp"],
            "ltq": tick["ltq"],
            "oi": tick["oi"]
        }
        df = pd.DataFrame([row])
        df.to_sql(
            self.table,
            self.engine,
            if_exists="append",
            index=False,
            method="multi"
        )
    def shutdown(self):
        # Placeholder for future thread-based queuing
        pass