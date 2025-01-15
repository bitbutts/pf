import subprocess
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os

# Run main.py if CSV does not exist
if not os.path.exists("xrp_token_payments.csv"):
    st.info("Generating transactions.csv using main.py...")
    result = subprocess.run(["python", "transaction_script.py"], check=True,text=True,capture_output=True)
    st.info(result.stdout)

# Function to load the CSV file
@st.cache_data
def load_csv():
    return pd.read_csv("xrp_token_payments.csv")

# Function to preprocess the data
def preprocess_data(df):
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    grouped_data = df.groupby('date').size().reset_index(name='count')
    return grouped_data

# Function to create the bar chart
def create_barchart(data):
    fig, ax = plt.subplots()
    ax.bar(data['date'], data['count'])
    ax.set_title("Count of Rows by Date")
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    return fig
    
#Function to plot the network graph    
def plot_graph(G):
    plt.figure(figsize=(12, 12))

    # Get node sizes and edge widths
    node_sizes = [G.nodes[node]['size'] * 10 for node in G.nodes]
    edge_widths = [G[u][v]['weight'] for u, v in G.edges]

    # Draw the graph
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="skyblue", alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

    plt.title("Network Graph of Address Relationships", fontsize=16)
    plt.axis("off")
    st.pyplot(plt)
    
# Streamlit app
st.title("Transaction Bar Chart")

# Load the CSV file
try:
    df = load_csv()
    grouped_data = preprocess_data(df)

    st.write("### Processed Data")
    st.dataframe(grouped_data)

    st.write("### Bar Chart")
    bar_chart = create_barchart(grouped_data)
    st.pyplot(bar_chart)

    st.write("## Network Graph of Address Relationships")
    G = create_graph(df)
    plot_graph(G)

except FileNotFoundError:
    st.error("The file 'transactions.csv' could not be found or generated.")
