# ml_tick_trainer/dhan_api_connector.py

import json
from dhanhq import DhanContext, dhanhq
from utils import log_text

def load_credentials():
    with open("ws.json", "r") as f:
        config = json.load(f)
    return config["client_id"], config["access_token"]

def connect():
    try:
        client_id, access_token = load_credentials()
        dhan_ctx = DhanContext(client_id, access_token)
        dhan = dhanhq(dhan_ctx)
        return dhan_ctx, dhan
    except Exception as e:
        log_text(f"Dhan connection failed: {e}")
        raise
