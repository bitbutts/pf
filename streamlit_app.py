import subprocess
subprocess.run(["pip", "install", "requests"])
subprocess.run(["pip", "install", "networkx"])
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os


# Run main.py if CSV does not exist
if not os.path.exists("xrp_token_payments.csv"):
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

    return {
        "Total Number of Addresses": total_addresses,
        "Unique 'to' Addresses": unique_to_addresses,
        "Unique 'from' Addresses": unique_from_addresses,
        "Mean Transaction Amount": mean_amount,
        "Median Transaction Amount": median_amount,
        "Minimum Transaction Amount": min_amount,
        "Maximum Transaction Amount": max_amount,
        "Standard Deviation of Amount": std_amount,
        "Total Transaction Volume": total_transaction_volume,
        "Total Transactions": total_transactions,
        "Earliest Transaction Date": earliest_transaction_date,
        "Latest Transaction Date": latest_transaction_date,
        "Total Transaction Days": total_transaction_days,
    }


# Function to create the bar chart
def create_barchart(data):
    fig, ax = plt.subplots()
    ax.bar(data['date'], data['count'])
    ax.set_title("Count of Rows by Date")
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
def plot_graph(G, node_scale=0.000008, edge_scale=0.000000003, spacing_factor=5):
    """
    Plots a network graph with adjustable node sizes, edge widths, and node spacing.

    Args:
        G (nx.DiGraph): The directed graph to plot.
        node_scale (float): Scaling factor for node sizes.
        edge_scale (float): Scaling factor for edge widths.
        spacing_factor (float): Multiplier to control the spacing between nodes.
    """
    plt.figure(figsize=(12, 12))

    # Get scaled node sizes and edge widths
    node_sizes = [G.nodes[node]['size'] * node_scale for node in G.nodes]
    edge_widths = [G[u][v]['weight'] * edge_scale for u, v in G.edges]

    # Calculate node positions with increased spacing
    pos = nx.spring_layout(G, seed=42, k=spacing_factor / len(G.nodes), iterations=50)

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="skyblue", alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5)

    plt.title("Network Graph of Address Relationships", fontsize=16)
    plt.axis("off")
    st.pyplot(plt)
    
# Streamlit app
st.title("PFT Transactions Last 30 days")

# Load the CSV file
try:
    df = load_csv()
    grouped_data = preprocess_data(df)

    st.write("### Statistical Overview")
    aggregates = calculate_aggregates(df)
    st.table(pd.DataFrame(aggregates, index=[0]))
    
    st.write("### Daily Transactions")
    bar_chart = create_barchart(grouped_data)
    st.pyplot(bar_chart)

    st.write("### Network Graph of Address Relationships")
    G = create_graph(df)
    plot_graph(G)

except FileNotFoundError:
    st.error("The file 'transactions.csv' could not be found or generated.")
