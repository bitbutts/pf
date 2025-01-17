import subprocess
subprocess.run(["pip", "install", "requests"])
subprocess.run(["pip", "install", "networkx"])
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import datetime


try:
    st.info("Running transaction_script.py...")
    result = subprocess.run(
        ["python", "transaction_script.py"],
        check=True,
        text=True,
        capture_output=True
    )
    st.success("Script ran successfully!")
    st.text("Output:\n" + result.stdout)
except subprocess.CalledProcessError as e:
    st.error("Script failed to run.")
    st.text(f"Error Output:\n{e.stderr}")

# Function to load the CSV file
@st.cache_data
def load_csv():
    return pd.read_csv("xrp_token_payments.csv")

# Function to preprocess the data
def preprocess_data(df):
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    grouped_data = df.groupby('date').size().reset_index(name='count')
    return grouped_data

def calculate_aggregates(df):
    total_addresses = len(set(df['to']).union(set(df['from'])))
    unique_to_addresses = df['to'].nunique()
    unique_from_addresses = df['from'].nunique()
    
    mean_amount = df['amount'].mean()
    median_amount = df['amount'].median()
    min_amount = df['amount'].min()
    max_amount = df['amount'].max()
    std_amount = df['amount'].std()
    total_transaction_volume = df['amount'].sum()
    
    total_transactions = len(df)
    earliest_transaction_date = pd.to_datetime(df['timestamp']).min().date()
    latest_transaction_date = pd.to_datetime(df['timestamp']).max().date()
    total_transaction_days = pd.to_datetime(df['timestamp']).dt.date.nunique()

    request_post_fiat_count = df[df['memo'].str.startswith("REQUEST_POST_FIAT", na=False)].shape[0]
    proposed_pf_count = df[df['memo'].str.startswith("PROPOSED PF", na=False)].shape[0]
    reward_response_count = df[df['memo'].str.startswith("REWARD RESPONSE", na=False)].shape[0]
    reward_response_sum = df[df['memo'].str.startswith("REWARD RESPONSE", na=False)]['amount'].sum()
    acceptance_reason_count = df[df['memo'].str.startswith("ACCEPTANCE REASON", na=False)].shape[0]
    initiation_reward_count = df[
        (df['from'] == 'r4yc85M1hwsegVGZ1pawpZPwj65SVs8PzD') &
        (~df['memo'].str.startswith("REQUEST_POST_FIAT", na=False)) &
        (~df['memo'].str.startswith("PROPOSED PF", na=False)) &
        (~df['memo'].str.startswith("REWARD RESPONSE", na=False)) &
        (~df['memo'].str.startswith("VERIFICATION PROMPT", na=False)) &
        (~df['memo'].str.startswith("Corbanu Reward", na=False)) &
        (~df['memo'].str.startswith("Initial PFT Grant Post Initiation", na=False)) &
        (df['amount'] <= 100)
    ]['to'].nunique()

    return {
        "ADDRESS COUNT": total_addresses,
        "TRANSACTION COUNT": total_transactions,
        "TRANSACTION VOLUME": total_transaction_volume,
        "MEAN TX VALUE": mean_amount,
        "INITIATIONS": initiation_reward_count,
        "PROPOSED TASKS": proposed_pf_count,
        "ACCEPTED TASKS": acceptance_reason_count,
        "COMPLETED TASKS": reward_response_count,
        "TASK REWARDS": reward_response_sum,

    }


# Function to create the bar chart
def create_barchart(data):
    fig, ax = plt.subplots()
    ax.bar(data['date'], data['count'])
    ax.set_title("Transaction Count by Date")
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    return fig


# Function to create the network graph
def create_graph(data):
    # Create a directed graph
    G = nx.DiGraph()

    # Calculate inflows for node size
    inflow = data.groupby('to')['amount'].sum()

    # Calculate edge thickness (transaction count)
    edge_weights = data.groupby(['from', 'to'])['amount'].sum()

    # Add all nodes with default sizes
    all_addresses = set(data['to']).union(set(data['from']))
    for address in all_addresses:
        G.add_node(address, size=inflow.get(address, 0))  # Default size to 0 if no inflow

    # Add edges with weights (transaction count)
    for (from_addr, to_addr), weight in edge_weights.items():
        G.add_edge(from_addr, to_addr, weight=weight)

    return G

# Function to plot the graph with adjustable node and edge sizes
def plot_graph(G, 
               min_node_size=100, max_node_size=2000,
               min_edge_size=1,   max_edge_size=8,
               spacing_factor=5):
    """
    Plots a network graph with automatically scaled node and edge sizes.

    Args:
        G (nx.DiGraph): The directed graph to plot.
        min_node_size (int): Minimum node size in the plotted graph.
        max_node_size (int): Maximum node size in the plotted graph.
        min_edge_size (int): Minimum edge width in the plotted graph.
        max_edge_size (int): Maximum edge width in the plotted graph.
        spacing_factor (float): Multiplier to control the spacing between nodes.
    """
    plt.figure(figsize=(12, 12))

    # Helper function to scale values (linear)
    def linear_scale(values, min_out, max_out):
        """
        Scales a list of numeric values to [min_out, max_out] linearly.
        If all values are the same, all get mid-range. If empty, return [].
        """
        if not values:
            return []
        min_val = min(values)
        max_val = max(values)
        if min_val == max_val:
            # If all values are the same, set them all to the midpoint.
            mid = (max_out + min_out) / 2
            return [mid] * len(values)
        # Otherwise, linear interpolation for each value:
        scaled = [
            min_out + (v - min_val) / (max_val - min_val) * (max_out - min_out)
            for v in values
        ]
        return scaled

    # 1) Collect node inflows and edge weights
    node_inflows = [G.nodes[node]['size'] for node in G.nodes]
    edge_wts = [G[u][v]['weight'] for u, v in G.edges]

    # 2) Auto-scale node sizes and edge widths
    scaled_node_sizes = linear_scale(node_inflows, min_node_size, max_node_size)
    scaled_edge_widths = linear_scale(edge_wts, min_edge_size, max_edge_size)

    # 3) Calculate node positions with spring layout
    pos = nx.spring_layout(G, seed=42, k=spacing_factor / max(len(G.nodes), 1), iterations=50)

    # 4) Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=scaled_node_sizes, node_color="skyblue", alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=scaled_edge_widths, alpha=0.5)

    plt.axis("off")
    st.pyplot(plt)

    
# Streamlit app
st.title("PFT Transactions Last 30 days")

# Load the CSV file
try:
    df = load_csv()
    
    # Convert timestamp column to datetime for filtering
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Calculate a default 30-day range ending yesterday
    default_end_date = datetime.date.today() - datetime.timedelta(days=1)
    default_start_date = default_end_date - datetime.timedelta(days=29)

    # Date selection with default values
    start_date = st.date_input(
        "Start Date",
        value=default_start_date,
        min_value=df['timestamp'].min().date(),
        max_value=df['timestamp'].max().date()
    )
    end_date = st.date_input(
        "End Date",
        value=default_end_date,
        min_value=df['timestamp'].min().date(),
        max_value=df['timestamp'].max().date()
    )

    # Validate date input
    if start_date > end_date:
        st.error("Start date must be before end date.")
    else:
        # Filter data within selected date range
        mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
        df_filtered = df[mask]

        # Preprocess and visualize
        grouped_data = preprocess_data(df_filtered)

        st.write("### Statistical Overview")
        aggregates = calculate_aggregates(df_filtered)
        st.table(pd.DataFrame(aggregates, index=[0]))

        st.write("### Initiations vs. Completed Tasks by Day")

        # 1) Filter Initiations (using your special condition)
        df_initiations = df_filtered[
        (df['from'] == 'r4yc85M1hwsegVGZ1pawpZPwj65SVs8PzD') &
        (~df['memo'].str.startswith("REQUEST_POST_FIAT", na=False)) &
        (~df['memo'].str.startswith("PROPOSED PF", na=False)) &
        (~df['memo'].str.startswith("REWARD RESPONSE", na=False)) &
        (~df['memo'].str.startswith("VERIFICATION PROMPT", na=False)) &
        (~df['memo'].str.startswith("Corbanu Reward", na=False)) &
        (~df['memo'].str.startswith("Initial PFT Grant Post Initiation", na=False)) &
        (df['amount'] <= 100)
    ]

        # Group initiations by day (distinct 'to')
        initiations_by_day = (
            df_initiations
            .groupby('date')['to']
            .nunique()
            .reset_index(name='initiations_count')
        )

        # 2) Filter Completed Tasks
        df_completed = df_filtered[df_filtered['memo'].str.startswith("REWARD RESPONSE", na=False)]

        # Group completed tasks by day (count how many)
        completed_by_day = (
            df_completed
            .groupby('date')
            .size()
            .reset_index(name='completed_count')
        )

        # 3) Merge both on date, fill missing days with 0
        df_line = pd.merge(
            initiations_by_day,
            completed_by_day,
            on='date',
            how='outer'
        ).fillna(0)

        # 4) Create one line chart with two lines
        df_line_chart = df_line.set_index('date')[['initiations_count', 'completed_count']]
        st.line_chart(data=df_line_chart, height=400)
        """
        st.write("### Daily Transactions")
        bar_chart = create_barchart(grouped_data)
        st.pyplot(bar_chart)
        """
        st.write("### Network Graph of Address Relationships")
        G = create_graph(df_filtered)
        plot_graph(G)

except FileNotFoundError:
    st.error("The file 'transactions.csv' could not be found or generated.")
