import json
import numpy as np
from datetime import date, datetime
from decimal import Decimal

import pandas as pd


class ExtendedEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()  # Convert to ISO 8601 string format
        elif isinstance(obj, date):
            return obj.isoformat()  # Convert to ISO 8601 string format for date objects
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super(ExtendedEncoder, self).default(obj)


def encode_records(df):
    records = df.to_dict("records")
    encoded_records = []
    for record in records:
        encoded_record = {}
        for key, value in record.items():
            if isinstance(value, Decimal):
                encoded_record[key] = str(value)
            elif isinstance(value, (datetime, pd.Timestamp, date)):
                encoded_record[key] = value.isoformat()
            elif pd.isna(value):  # Check if the value is NaN or None
                encoded_record[key] = None  # Replace NaN with None, which will be null in JSON
            else:
                encoded_record[key] = value
        encoded_records.append(encoded_record)
    return encoded_records
