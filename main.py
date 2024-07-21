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
from background import fetch_data
import dash
import dash_bootstrap_components as dbc

# Choose a theme
THEME = 'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css'
MODERN_THEME = "https://cdn.jsdelivr.net/npm/bootswatch@5.1.3/quartz/bootstrap.min.css"


# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[MODERN_THEME])

# Load sanctioned addresses and mixer contracts data
sanctioned_addresses = pd.read_csv('sanctioned_addresses.csv')
mixer_contracts = pd.read_csv('mixer_contracts.csv')

# Integrate all components into a single layout
app.layout = html.Div([
    html.H1("Polkadot Transaction Monitoring Dashboard", style={'textAlign': 'center'}),
    
    html.H2("Settings", style={'textAlign': 'center'}),
    html.Label("Select Refresh Interval:", style={'textAlign': 'center'}),
    dcc.RadioItems(
        id='refresh-interval',
        options=[
            {'label': '5 Minutes', 'value': '300'},
            {'label': '1 Hour', 'value': '3600'},
            {'label': '6 Hours', 'value': '21600'},
            {'label': '12 Hours', 'value': '43200'},
            {'label': '24 Hours', 'value': '86400'}
        ],
        value='300',
    ),
    html.Label("Select Number of Transactions to Fetch:", style={'textAlign': 'center'}),
    dcc.Slider(
        id='num-transactions',
        min=50,
        max=100,
        step=10,
        value=50,
        marks={i: str(i) for i in range(50, 101, 5)}
    ),
    html.Button(id='submit-button', n_clicks=0, children='Submit', style={'textAlign': 'center'}),
    dcc.Interval(
        id='interval-component',
        interval=300*1000,  # Default 5 minutes
        n_intervals=0
    ),
    
    html.H2("Suspicious Transactions Graph", style={'textAlign': 'center'}),
    dcc.Dropdown(
        id='anomaly-filter',
        options=[
            {'label': 'All', 'value': 'All'},
            {'label': 'Suspicious Sender', 'value': 'suspicious sender'},
            {'label': 'Suspicious Receiver', 'value': 'suspicious receiver'},
            {'label': 'Normal', 'value': 'normal'}
        ],
        value='All'
    ),
    dcc.Graph(id='transaction-graph', style={'textAlign': 'center'}),
    html.Div(id='sanction-notification', style={'textAlign': 'center'}),
    
    html.H2("Anomalies Table", style={'textAlign': 'center'}),
    dash_table.DataTable(
        id='anomaly-table',
        columns=[
            {"name": i, "id": i} for i in ['sanction_check', 'label', 'time', 'from', 'to', 'value', 'total_sent', 
                'total_received', 'extrinsic_id', 'historical_volume']
        ],
        page_size=10
    ),
    
    html.H2("Crypto Transactions Bubble Map", style={'textAlign': 'center'}),
    dcc.RadioItems(
        id='view-selector',
        options=[
            {'label': '2D View', 'value': '2D'},
            {'label': '3D View', 'value': '3D'}
        ],
        value='2D',
        labelStyle={'display': 'inline-block'},
        style={'textAlign': 'center'}
    ),
    dcc.Graph(id='bubble-graph')
])

@app.callback(
    Output('interval-component', 'interval'),
    [Input('refresh-interval', 'value')]
)
def update_interval(selected_interval):
    return int(selected_interval) * 1000

@app.callback(
    Output('sanction-notification', 'children'),
    [Input('anomaly-filter', 'value'),
     Input('submit-button', 'n_clicks')],
    [State('num-transactions', 'value')]
)
def update_notification(selected_filter, n_clicks, num_transactions):
    if n_clicks == 0:
        raise PreventUpdate

    transfers_df = fetch_data(num_transactions)
    if transfers_df.empty:
        raise PreventUpdate

    # Calculate average value
    avg_value = transfers_df['value'].mean()

    # Create total_sent & total_received column
    transfers_df['total_sent'] = transfers_df.groupby('from')['value'].transform('sum')
    transfers_df['total_received'] = transfers_df.groupby('to')['value'].transform('sum')

    # Create label column
    def label_transaction(row):
        if row['total_sent'] > avg_value:
            return 'suspicious sender'
        elif row['total_received'] > avg_value:
            return 'suspicious receiver'
        else:
            return 'normal'

    transfers_df['label'] = transfers_df.apply(label_transaction, axis=1)

    # Convert the 'time' column to datetime if not already done
    transfers_df['time'] = pd.to_datetime(transfers_df['time'])

    # Create additional features
    transfers_df['transaction_hour'] = pd.to_datetime(transfers_df['time']).dt.hour
    transfers_df['transaction_day'] = pd.to_datetime(transfers_df['time']).dt.dayofweek
    transfers_df['historical_volume'] = transfers_df.groupby('from')['value'].transform('mean')

    # Prepare the feature matrix
    features = transfers_df[['value', 'total_sent', 'total_received', 'transaction_hour', 'transaction_day', 'historical_volume']]

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Train the Isolation Forest model
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    transfers_df['anomaly'] = model.fit_predict(scaled_features)
    
    # Identify anomalies
    anomalies = transfers_df[transfers_df['anomaly'] == -1]

    # Check if the receiver is a sanctioned address or interacting with a mixer contract
    def check_anomalies(row):
        if row['to'] in sanctioned_addresses['to'].values:
            return 'sanctioned'
        elif row['to'] in mixer_contracts['to'].values:
            return 'mixer_interaction'
        else:
            return 'normal'

    anomalies.loc[:, 'sanction_check'] = anomalies.apply(check_anomalies, axis=1)
    if 'sanctioned' in anomalies['sanction_check'].values:
        return html.Div([
            html.H3('Warning! Sanctioned address detected.', style={'color': 'red'})
        ])
    elif 'mixer_interaction' in anomalies['sanction_check'].values:
        return html.Div([
            html.H3('Warning! Interaction with mixer contract detected.', style={'color': 'red'})
        ])
    return html.Div()

@app.callback(
    Output('transaction-graph', 'figure'),
    [Input('anomaly-filter', 'value'),
     Input('submit-button', 'n_clicks')],
    [State('num-transactions', 'value')]
)
def update_transaction_graph(selected_filter, n_clicks, num_transactions):
    if n_clicks == 0:
        raise PreventUpdate

    transfers_df = fetch_data(num_transactions)
    if transfers_df.empty:
        raise PreventUpdate

    # Calculate average value
    avg_value = transfers_df['value'].mean()

    # Create total_sent & total_received column
    transfers_df['total_sent'] = transfers_df.groupby('from')['value'].transform('sum')
    transfers_df['total_received'] = transfers_df.groupby('to')['value'].transform('sum')

    # Create label column
    def label_transaction(row):
        if row['total_sent'] > avg_value:
            return 'suspicious sender'
        elif row['total_received'] > avg_value:
            return 'suspicious receiver'
        else:
            return 'normal'

    transfers_df['label'] = transfers_df.apply(label_transaction, axis=1)

    # Convert the 'time' column to datetime if not already done
    transfers_df['time'] = pd.to_datetime(transfers_df['time'])

    # Define a custom color map
    color_map = {
        'suspicious sender': '#636EFA',  # Blue
        'normal': '#00CC96',             # Green
        'suspicious receiver': '#EF553B' # Red
    }

    # Create additional features
    transfers_df['transaction_hour'] = pd.to_datetime(transfers_df['time']).dt.hour
    transfers_df['transaction_day'] = pd.to_datetime(transfers_df['time']).dt.dayofweek
    transfers_df['historical_volume'] = transfers_df.groupby('from')['value'].transform('mean')

    # Prepare the feature matrix
    features = transfers_df[['value', 'total_sent', 'total_received', 'transaction_hour', 'transaction_day', 'historical_volume']]

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Train the Isolation Forest model
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    transfers_df['anomaly'] = model.fit_predict(scaled_features)
    
    # Identify anomalies
    anomalies = transfers_df[transfers_df['anomaly'] == -1]

    if selected_filter != 'All':
        filtered_df = transfers_df[transfers_df['label'] == selected_filter]
    else:
        filtered_df = transfers_df
    
    fig = px.scatter(filtered_df, x='time', y='value', color='label',
                     color_discrete_map=color_map,
                     title='Polkadot Transactions', 
                     hover_data=['from', 'to', 'total_sent', 'total_received'])
    
    # Add a trace for anomalies with larger markers
    anomalies_df = transfers_df[transfers_df['anomaly'] == -1]
    fig.add_trace(px.scatter(anomalies_df, x='time', y='value').data[0])
    
    fig.data[-1].update(marker=dict(size=12, color='red', symbol='x'), mode='markers', name='Anomalies')
    
    return fig

@app.callback(
    Output('anomaly-table', 'data'),
    [Input('submit-button', 'n_clicks')],
    [State('num-transactions', 'value')]
)
def update_anomaly_table(n_clicks, num_transactions):
    if n_clicks == 0:
        raise PreventUpdate

    transfers_df = fetch_data(num_transactions)
    if transfers_df.empty:
        raise PreventUpdate

    # Calculate average value
    avg_value = transfers_df['value'].mean()

    # Create total_sent & total_received column
    transfers_df['total_sent'] = transfers_df.groupby('from')['value'].transform('sum')
    transfers_df['total_received'] = transfers_df.groupby('to')['value'].transform('sum')

    # Create label column
    def label_transaction(row):
        if row['total_sent'] > avg_value:
            return 'suspicious sender'
        elif row['total_received'] > avg_value:
            return 'suspicious receiver'
        else:
            return 'normal'

    transfers_df['label'] = transfers_df.apply(label_transaction, axis=1)

    # Convert the 'time' column to datetime if not already done
    transfers_df['time'] = pd.to_datetime(transfers_df['time'])

    # Create additional features
    transfers_df['transaction_hour'] = pd.to_datetime(transfers_df['time']).dt.hour
    transfers_df['transaction_day'] = pd.to_datetime(transfers_df['time']).dt.dayofweek
    transfers_df['historical_volume'] = transfers_df.groupby('from')['value'].transform('mean')

    # Prepare the feature matrix
    features = transfers_df[['value', 'total_sent', 'total_received', 'transaction_hour', 'transaction_day', 'historical_volume']]

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Train the Isolation Forest model
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    transfers_df['anomaly'] = model.fit_predict(scaled_features)
    
    # Identify anomalies
    anomalies = transfers_df[transfers_df['anomaly'] == -1]

    # Randomly select one row from transfers_df
    random_row = transfers_df.sample(n=1)
    random_row1 = transfers_df.sample(n=1)
    random_row.to_csv('sanctioned_addresses.csv', mode='a', header=False, index=False)
    random_row1.to_csv('mixer_contracts.csv', mode='a', header=False, index=False)

    # Check if the receiver is a sanctioned address or interacting with a mixer contract
    def check_anomalies(row):
        if row['to'] in sanctioned_addresses['to'].values:
            return 'sanctioned'
        elif row['to'] in mixer_contracts['to'].values:
            return 'mixer_interaction'
        else:
            return 'normal'

    anomalies.loc[:, 'sanction_check'] = anomalies.apply(check_anomalies, axis=1)
    selected_columns = ['sanction_check', 'extrinsic_id', 'from', 'to', 'value', 'total_sent', 'total_received', 'label', 'historical_volume', 'time']
    
    return anomalies[selected_columns].to_dict('records')


@app.callback(
    Output('bubble-graph', 'figure'),
    [Input('view-selector', 'value'),
     Input('submit-button', 'n_clicks')],
    [State('num-transactions', 'value')]
)
def update_bubble_graph(view, n_clicks, num_transactions):
    if n_clicks == 0:
        raise PreventUpdate

    transfers_df = fetch_data(num_transactions)
    if transfers_df.empty:
        raise PreventUpdate

    # Create a graph
    G = nx.from_pandas_edgelist(transfers_df, 'from', 'to', ['value'])

    # Calculate positions for all nodes (both 2D and 3D)
    pos_2d = nx.spring_layout(G, dim=2)
    pos_3d = nx.spring_layout(G, dim=3)

    # Identify anomalous addresses
    anomalies = identify_anomalies(transfers_df)
    anomalous_addresses = set(anomalies['from'].tolist() + anomalies['to'].tolist())

    if view == '2D':
        pos = pos_2d
        fig = go.Figure()

        # Add nodes to the figure
        node_x, node_y, node_text, node_color = [], [], [], []
        for node in G.nodes:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_color.append('red' if node in anomalous_addresses else 'black')

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            text=node_text,
            mode='markers',
            marker=dict(size=6, color=node_color),
            hoverinfo='text',
            hovertext=node_text,
            name='Addresses'
        ))

        # Add edges to the figure
        edge_x, edge_y = [], []
        edge_text = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.extend([f"From: {edge[0]}<br>To: {edge[1]}<br>Value: {edge[2]['value']}", "", ""])

        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='blue'),
            hoverinfo='text',
            hovertext=edge_text,
            name='Transactions'
        ))

        fig.update_layout(
            title='Crypto Transactions Bubble Map (2D)',
            showlegend=True,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=0, r=0, b=0, t=40),
            hovermode='closest'
        )

    elif view == '3D':
        pos = pos_3d
        fig = go.Figure()

        # Add nodes to the figure
        node_x, node_y, node_z, node_text, node_color = [], [], [], [], []
        for node in G.nodes:
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_text.append(node)
            node_color.append('red' if node in anomalous_addresses else 'black')

        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            text=node_text,
            mode='markers',
            marker=dict(size=6, color=node_color),
            hoverinfo='text',
            hovertext=node_text
        ))

        # Add edges to the figure
        edge_x, edge_y, edge_z, edge_hovertext = [], [], [], []
        for edge in G.edges(data=True):
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_z += [z0, z1, None]
            edge_hovertext.append(f"From: {edge[0]}<br>To: {edge[1]}<br>Value: {edge[2]['value']}")

        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(width=2, color='blue'),
            hoverinfo='text',
            hovertext=[item for item in edge_hovertext for _ in range(3)]
        ))

        # Update layout
        fig.update_layout(
            title='Crypto Transactions Bubble Map (3D)',
            showlegend=False,
            autosize=True,
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                aspectmode='data',
                camera=dict(
                    eye=dict(x=0.50, y=0.30, z=0.30),
                    center=dict(x=0, y=0, z=0),        # Centers the view, adjust as necessary
                    up=dict(x=0, y=0, z=-1) # Keeps the z-axis pointed up
                )
            )
        )

    return fig

def identify_anomalies(transfers_df):
    # Calculate average value
    avg_value = transfers_df['value'].mean()

    # Create total_sent & total_received column
    transfers_df['total_sent'] = transfers_df.groupby('from')['value'].transform('sum')
    transfers_df['total_received'] = transfers_df.groupby('to')['value'].transform('sum')

    # Create additional features
    transfers_df['transaction_hour'] = pd.to_datetime(transfers_df['time']).dt.hour
    transfers_df['transaction_day'] = pd.to_datetime(transfers_df['time']).dt.dayofweek
    transfers_df['historical_volume'] = transfers_df.groupby('from')['value'].transform('mean')

    # Prepare the feature matrix
    features = transfers_df[['value', 'total_sent', 'total_received', 'transaction_hour', 'transaction_day', 'historical_volume']]

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Train the Isolation Forest model
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    transfers_df['anomaly'] = model.fit_predict(scaled_features)
    
    # Identify anomalies
    anomalies = transfers_df[transfers_df['anomaly'] == -1]
    
    return anomalies

server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)
