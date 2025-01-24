import subprocess
import requests
import networkx
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import datetime

conn = st.connection("neon", type="sql")

def fetch_data():
    try:
        # Example query: replace `your_table` with an actual table
        query = "SELECT * FROM pft_transactions;"
        
        # Execute the query; returns a list of dicts by default
        result = conn.query(query, ttl=600)  
        
        # Convert the result to a pandas DataFrame
        df = pd.DataFrame(result)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Function to preprocess the data
def preprocess_data(df):
    df['date'] = pd.to_datetime(df['transaction_timestamp']).dt.date
    grouped_data = df.groupby('date').size().reset_index(name='count')
    return grouped_data

def calculate_aggregates(df):
    total_addresses = len(set(df['to_address']).union(set(df['from_address'])))
    unique_to_addresses = df['to_address'].nunique()
    unique_from_addresses = df['from_address'].nunique()
    
    mean_amount = int(round(df['amount'].mean()))
    median_amount = df['amount'].median()
    min_amount = df['amount'].min()
    max_amount = df['amount'].max()
    std_amount = df['amount'].std()
    total_transaction_volume = int(round(df['amount'].sum()))
    
    total_transactions = len(df)

    request_post_fiat_count = df[df['memo'].str.startswith("REQUEST_POST_FIAT", na=False)].shape[0]
    proposed_pf_count = df[df['memo'].str.startswith("PROPOSED PF", na=False)].shape[0]
    reward_response_count = df[(df['memo'].str.startswith("REWARD RESPONSE", na=False) |
         df['memo'].str.startswith("Corbanu Reward", na=False))].shape[0]
    reward_response_sum = int(round(df[(df['memo'].str.startswith("REWARD RESPONSE", na=False) |
         df['memo'].str.startswith("Corbanu Reward", na=False))]['amount'].sum()))
    acceptance_reason_count = df[df['memo'].str.startswith("ACCEPTANCE REASON", na=False)].shape[0]
    initiation_reward_count = df[
        (df['from_address'] == 'r4yc85M1hwsegVGZ1pawpZPwj65SVs8PzD') &
        (~df['memo'].str.startswith("REQUEST_POST_FIAT", na=False)) &
        (~df['memo'].str.startswith("PROPOSED PF", na=False)) &
        (~df['memo'].str.startswith("REWARD RESPONSE", na=False)) &
        (~df['memo'].str.startswith("VERIFICATION PROMPT", na=False)) &
        (~df['memo'].str.startswith("Corbanu Reward", na=False)) &
        (~df['memo'].str.startswith("Initial PFT Grant Post Initiation", na=False)) &
        (df['amount'] <= 100)
    ]['to_address'].nunique()

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
    
def calculate_tasks(df):
    # 1) Filter Initiations
    df_initiations = df[
        (df['from_address'] == 'r4yc85M1hwsegVGZ1pawpZPwj65SVs8PzD') &
        (~df['memo'].str.startswith("REQUEST_POST_FIAT", na=False)) &
        (~df['memo'].str.startswith("PROPOSED PF", na=False)) &
        (~df['memo'].str.startswith("REWARD RESPONSE", na=False)) &
        (~df['memo'].str.startswith("VERIFICATION PROMPT", na=False)) &
        (~df['memo'].str.startswith("Corbanu Reward", na=False)) &
        (~df['memo'].str.startswith("Initial PFT Grant Post Initiation", na=False)) &
        (df['amount'] <= 100)
    ]

    # Group initiations by day (distinct 'to_address')
    initiations_by_day = (
        df_initiations
        .groupby('date')['to_address']
        .nunique()
        .reset_index(name='initiations_count')
    )

    # 2) Filter Completed Tasks
    df_completed = df[(df['memo'].str.startswith("REWARD RESPONSE", na=False) |
         df['memo'].str.startswith("Corbanu Reward", na=False))]

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
    return df_line_chart

def calculate_leaderboard(df, from_address):
    """
    Generate a leaderboard of addresses with the highest sum of amounts and their counts.
    """
    # Filter rows matching the conditions
    df_filtered = df[
        (df['from_address'] == from_address) &
        (df['memo'].str.startswith("REWARD RESPONSE", na=False) |
         df['memo'].str.startswith("Corbanu Reward", na=False))
    ]

    # Group by 'to_address', summing 'amount' and counting occurrences
    leaderboard = (
        df_filtered
        .groupby('to_address')
        .agg(
            awarded_PFT=('amount', lambda x: round(x.sum())),  # Round the sum
            completed_tasks=('amount', 'count')
        )
        .reset_index()
        .sort_values(by='awarded_PFT', ascending=False)
        .head(10)
    )

    return leaderboard


def calculate_amount_by_day(df, from_address):
    """
    Create a line chart data for the sum of amounts by day, separated by memo type.
    
    Args:
        df (pd.DataFrame): The prefiltered DataFrame containing transaction data.
        from_address (str): The address to filter transactions by.
    
    Returns:
        pd.DataFrame: Pivoted DataFrame with dates as index and memo types as columns.
    """
    # Ensure 'date' is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    
    # Filter rows matching the conditions
    df_filtered = df[
        (df['from_address'] == from_address) &
        (
            df['memo'].str.startswith("REWARD RESPONSE", na=False) |
            df['memo'].str.startswith("Corbanu Reward", na=False)
        )
    ].copy()
    
    # Create a new column for memo type
    df_filtered['memo_type'] = df_filtered['memo'].apply(
        lambda x: "TASKBOT" if x.startswith("REWARD RESPONSE") else "CORBANU"
    )
    
    # Group by 'date' and 'memo_type', summing 'amount'
    day_line = (
        df_filtered
        .groupby(['date', 'memo_type'])
        .agg(total_amount=('amount', 'sum'))
        .reset_index()
    )
    
    # Pivot the DataFrame to have separate columns for each memo_type
    amount_by_day = day_line.pivot(index='date', columns='memo_type', values='total_amount').fillna(0)
    
    # Sort by date
    amount_by_day = amount_by_day.sort_index()
    
    return amount_by_day

# Function to create the bar chart
def create_barchart(data):
    fig, ax = plt.subplots()
    ax.bar(data['date'], data['count'])
    ax.set_title("Transaction Count by Date")
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    return fig


def set_node_highlights(G, highlight_dict):
    """
    For each node in highlight_dict, set the given attributes (e.g. color, label).
    highlight_dict looks like:
        {
          'some_node': {'color': 'blue', 'label': 'my_label'},
          ...
        }
    """
    for node, props in highlight_dict.items():
        if node in G:  # Only set if the node actually exists in the graph
            for key, value in props.items():
                G.nodes[node][key] = value

def create_graph(data, highlight_dict=None):
    # Create a directed graph
    G = nx.DiGraph()

    # Calculate inflows for node size
    inflow = data.groupby('to_address')['amount'].sum()

    # Calculate edge thickness (transaction count)
    edge_weights = data.groupby(['from_address', 'to_address'])['amount'].sum()

    # Add all nodes with default sizes
    all_addresses = set(data['to_address']).union(set(data['from_address']))
    for address in all_addresses:
        G.add_node(address, size=inflow.get(address, 0))  # default node size

    # Add edges with weights
    for (from_addr, to_addr), weight in edge_weights.items():
        G.add_edge(from_addr, to_addr, weight=weight)

    # If we have a highlight dictionary, set those node attributes
    if highlight_dict:
        set_node_highlights(G, highlight_dict)

    return G

def plot_graph(G, 
               min_node_size=100, max_node_size=2000,
               min_edge_size=1,   max_edge_size=8,
               spacing_factor=5,
               default_color="skyblue",
               default_label="default",
               default_size_override=None):
    """
    Plots a network graph with automatically scaled node and edge sizes.
    - If a node has 'size_override', we use that instead of the scaled size.
    - Groups nodes by (color, label) so that each group is drawn separately,
      allowing a color legend via plt.legend().
    """

    plt.figure(figsize=(12, 12))

    def linear_scale(values, min_out, max_out):
        if not values:
            return []
        min_val = min(values)
        max_val = max(values)
        if min_val == max_val:
            mid = (max_out + min_out) / 2
            return [mid] * len(values)
        return [
            min_out + (v - min_val) / (max_val - min_val) * (max_out - min_out)
            for v in values
        ]

    # -- 1) Scale Node & Edge Sizes --
    node_list = list(G.nodes())
    node_inflows = [G.nodes[n].get("size", 0) for n in node_list]
    edge_wts     = [G[u][v]["weight"] for u, v in G.edges]

    scaled_node_sizes  = linear_scale(node_inflows, min_node_size, max_node_size)
    scaled_edge_widths = linear_scale(edge_wts,     min_edge_size, max_edge_size)

    # -- 2) Apply size_override if present --
    for i, node in enumerate(node_list):
        override = G.nodes[node].get("size_override", default_size_override)
        if override is not None:
            scaled_node_sizes[i] = override

    # -- 3) Layout --
    pos = nx.spring_layout(
        G, 
        seed=42,
        k=spacing_factor / max(len(G.nodes), 1),
        iterations=50
    )

    # -- 4) Draw Edges Once --
    nx.draw_networkx_edges(
        G, pos,
        width=scaled_edge_widths,
        alpha=0.5,
        arrows=True,
        arrowstyle="-|>"
    )

    # -- 5) Group nodes by (color, label) so we can draw them separately --
    #         This allows us to pass a distinct 'label' to each color group,
    #         so that plt.legend() will show them.
    from collections import defaultdict
    color_label_groups = defaultdict(list)  # (color, label) -> list of indices

    for i, node in enumerate(node_list):
        c = G.nodes[node].get("color", default_color)
        lbl = G.nodes[node].get("label", default_label)
        color_label_groups[(c, lbl)].append(i)

    # Now we draw each group of nodes separately
    for (c, lbl), indices in color_label_groups.items():
        # Extract the relevant node indices
        group_node_list   = [node_list[i] for i in indices]
        group_node_sizes  = [scaled_node_sizes[i] for i in indices]

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=group_node_list,
            node_size=group_node_sizes,
            node_color=c,
            alpha=0.8,
            label=lbl  # <-- This is how we get a legend entry
        )


    plt.legend(scatterpoints=1, loc="best")
    
    plt.axis("off")
    fig = plt.gcf()
    plt.show(fig)



    
# Streamlit app
st.title("PFT Transactions Last 30 days")

# Load the CSV file
try:
    df = fetch_data()
    
    # Convert transaction_timestamp column to datetime for filtering
    df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'])

    # Calculate a default 30-day range ending yesterday
    default_end_date = datetime.date.today() - datetime.timedelta(days=1)
    default_start_date = default_end_date - datetime.timedelta(days=29)
    
    # Date selection with default values
    start_date = st.date_input(
        "Start Date",
        value=default_start_date,
        min_value=df['transaction_timestamp'].min().date(),
        max_value=df['transaction_timestamp'].max().date()
    )
    end_date = st.date_input(
        "End Date",
        value=default_end_date,
        min_value=df['transaction_timestamp'].min().date(),
        max_value=df['transaction_timestamp'].max().date()
    )

    # Validate date input
    if start_date > end_date:
        st.error("Start date must be before end date.")
    else:
        # Filter data within selected date range
        mask = (df['transaction_timestamp'].dt.date >= start_date) & (df['transaction_timestamp'].dt.date <= end_date)
        df_filtered = df[mask]
        from_address = 'r4yc85M1hwsegVGZ1pawpZPwj65SVs8PzD'
        # Preprocess and visualize
        grouped_data = preprocess_data(df_filtered)

        st.write("### Statistical Overview")
        aggregates = calculate_aggregates(df_filtered)
        st.table(pd.DataFrame(aggregates, index=[0]))

        st.write("### Initiations vs. Completed Tasks by Day")
        task_data = calculate_tasks(df_filtered)
            
        st.line_chart(data=task_data, height=400)
        # Generate the leaderboard table
        leaderboard = calculate_leaderboard(df_filtered, from_address)
        st.write("### Leaderboard")
        st.table(leaderboard)

        amount_by_day = calculate_amount_by_day(df_filtered, from_address)
        st.write("### Daily Earned PFT (Taskbot + Corbanu)")
        st.line_chart(data=amount_by_day, height=400)
        highlight_dict = {"r4yc85M1hwsegVGZ1pawpZPwj65SVs8PzD": {"color": "green",  "label": "task_node","size_override": 200},}
        st.write("### Network Graph of Address Relationships")

        G = create_graph(df_filtered, highlight_dict=highlight_dict)

        # Plot the graph
        plot_graph(G, 
                   min_node_size=50, max_node_size=2000,
                   min_edge_size=0.2,   max_edge_size=4,
                   spacing_factor=900)

except FileNotFoundError:
    st.error("The file 'transactions.csv' could not be found or generated.")
