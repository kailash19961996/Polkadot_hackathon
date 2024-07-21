import requests
import pandas as pd
import time
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash import dash_table
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import networkx as nx
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

# 1. Getting the data

API_KEY = '1c269c1797c54f31b3e93a85ad492dd8'
SUBSCAN_API_URL = 'https://polkadot.api.subscan.io/api/v2/scan/transfers'

# Headers for the API request
headers = {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY
}

def fetch_data(row_count):
    params = {
        "row": row_count,  # Number of transfers to fetch
        "page": 0    # Page number
    }

    try:
        response = requests.post(SUBSCAN_API_URL, headers=headers, json=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get('code') == 0:
            transfers = data['data']['transfers']
            
            transfers_list = []
            for tx in transfers:
                transfer = {
                    'extrinsic_id': tx.get('extrinsic_index'),
                    'block': tx.get('block_num'),
                    'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(tx.get('block_timestamp'))),
                    'from': tx.get('from'),
                    'to': tx.get('to'),
                    'value': float(tx.get('amount', 0)) / 1e10,  # Convert Planck to DOT
                    'result': 'Success' if tx.get('success') else 'Failed'
                }
                transfers_list.append(transfer)
            
            transfers_df = pd.DataFrame(transfers_list)
            return transfers_df
        else:
            print("Failed to fetch data:", data.get('message', 'Unknown error'))
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        if hasattr(e, 'response'):
            print(f"Response content: {e.response.text}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()

