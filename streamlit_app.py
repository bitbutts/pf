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
from collections import defaultdict

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


def calculate_transaction_count(df):
    """
    Groups transactions by date and counts how many transactions occurred on each date.
    Returns a DataFrame with 'date' as the index and a single column 'transaction_count'.
    """
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    # Group by date, count number of transactions
    df_counts = (
        df.groupby('date')
          .size()  # Count rows per date
          .reset_index(name='transaction_count')
    )
    
    # Set 'date' as the index so we can easily plot against it
    df_counts.set_index('date', inplace=True)
    
    return df_counts
    
def create_barchart(df_counts):
    """
    Takes a DataFrame with index = date and a column = 'transaction_count',
    and creates a bar chart of transaction counts by date.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot a simple bar chart using the index as X values
    ax.bar(df_counts.index, df_counts['transaction_count'], color='skyblue')

    ax.set_title("Transaction Count by Date")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Transactions")
    plt.xticks(rotation=45)

    # Prevent label overlap
    fig.tight_layout()
    
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
    #inflow = data.groupby('to_address')['amount'].sum()
    inflow_by_address = data.groupby("to_address")["amount"].sum()

    # Sum of amounts going out of each address
    outflow_by_address = data.groupby("from_address")["amount"].sum()
    # Calculate edge thickness (transaction count)
    edge_weights = data.groupby(['from_address', 'to_address'])['amount'].count()

    # Add all nodes with default sizes
    all_addresses = set(data['to_address']).union(set(data['from_address']))
    for address in all_addresses:
        in_ = inflow_by_address.get(address, 0)
        out_ = outflow_by_address.get(address, 0)
        net_flow = in_ - out_
    
        # Set net_flow as the node "size"
        G.add_node(address, size=net_flow)  # default node size
        if abs(net_flow) > 2_000_000:
            G.nodes[address]['color'] = 'grey'
            G.nodes[address]['size_override'] = 200
            G.nodes[address]['label'] = ""

    # Add edges with weights
    for (from_addr, to_addr), weight in edge_weights.items():
        G.add_edge(from_addr, to_addr, weight=weight)

    # If we have a highlight dictionary, set those node attributes
    if highlight_dict:
        set_node_highlights(G, highlight_dict)

    return G

def plot_graph(
    G, 
    min_node_size, 
    max_node_size,
    min_edge_size,   
    max_edge_size,
    spacing_factor,
    default_color="skyblue",
    default_label="default"
):
    """
    Plots a network graph in Matplotlib, using:
      1) "size_override" (if present) before scaling,
      2) Groups nodes by (color, label) for multiple draw calls,
         enabling a legend via plt.legend().

    1) We skip overridden nodes from the min/max scaling calculation,
       so their large (or small) override doesn't distort the range 
       for normal nodes.
    2) All normal (non-overridden) nodes get scaled to [min_node_size, max_node_size].
    3) For the legend, we do one draw call per (color, label) group.
    """
    plt.figure(figsize=(15, 15))

    def linear_scale(values, min_out, max_out):
        if not values:
            return []
    
    # 1) Clamp all negatives to 0 (so the smallest we'll see is 0)
        clamped = [max(0, v) for v in values]
    
    # 2) Now find min and max from the clamped set
        min_val, max_val = min(clamped), max(clamped)

        if min_val == max_val:
            # All identical => just give them the midpoint
            mid = (max_out + min_out) / 2
            return [mid] * len(clamped)
    
        # 3) Scale to [min_out, max_out]
        return [
            min_out + (v - min_val) / (max_val - min_val) * (max_out - min_out)
            for v in clamped
        ]
    
    # 1) Collect nodes, edges
    node_list = list(G.nodes())
    edge_wts  = [G[u][v]["weight"] for u, v in G.edges]

    # 2) Prepare two lists: normal vs. overridden
    raw_sizes = [G.nodes[n].get("size", 0) for n in node_list]
    overrides = [G.nodes[n].get("size_override") for n in node_list]

    normal_nodes   = []
    normal_values  = []
    override_nodes = {}

    for i, node in enumerate(node_list):
        if overrides[i] is not None:
            # This node has a size_override
            override_nodes[node] = overrides[i]
        else:
            # We'll scale this node normally
            normal_nodes.append(node)
            normal_values.append(raw_sizes[i])

    # 3) Scale the normal nodes only
    scaled_normal_sizes = linear_scale(normal_values, min_node_size, max_node_size)

    # Create final array of node sizes (same length as node_list)
    scaled_node_sizes = [0] * len(node_list)

    # Fill in the scaled sizes for normal nodes
    normal_idx = 0
    for i, node in enumerate(node_list):
        if node not in override_nodes:
            scaled_node_sizes[i] = scaled_normal_sizes[normal_idx]
            normal_idx += 1

    # Fill in override sizes
    for i, node in enumerate(node_list):
        if node in override_nodes:
            scaled_node_sizes[i] = override_nodes[node]

    # 4) Scale edge widths
    scaled_edge_widths = linear_scale(edge_wts, min_edge_size, max_edge_size)

    # 5) Layout
    pos = nx.spring_layout(
        G, 
        seed=42,
        k=spacing_factor / max(len(G.nodes), 1),
        iterations=50
    )

    # Draw edges once
    nx.draw_networkx_edges(
        G, pos,
        width=scaled_edge_widths,
        alpha=0.5,
        arrows=True,
        arrowstyle="-|>"
    )

    # 6) Group nodes by (color, label) => one draw call per group => legend
    color_label_groups = defaultdict(list)  # (color, label) -> list of node indices
    for i, node in enumerate(node_list):
        color = G.nodes[node].get("color", default_color)
        lbl   = G.nodes[node].get("label", default_label)
        color_label_groups[(color, lbl)].append(i)

    # For each group, draw the subset of nodes
    for (color, lbl), indices in color_label_groups.items():
        group_node_list  = [node_list[i] for i in indices]
        group_node_sizes = [scaled_node_sizes[i] for i in indices]

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=group_node_list,
            node_size=group_node_sizes,
            node_color=color,
            alpha=0.8,
            label=lbl  # this is used in the legend
        )

    # If you want text labels, you can do them separately
    # text_labels = {
    #     n: G.nodes[n].get("label", str(n)) 
    #     for n in node_list 
    #     if "label" in G.nodes[n]
    # }
    # nx.draw_networkx_labels(G, pos, labels=text_labels, font_size=10)

    plt.legend(scatterpoints=1, loc="best")
    plt.axis("off")
    st.pyplot(plt)





# Streamlit app
st.title("PFT Transactions Overview")
twit = "https://x.com/pftdaily"
st.markdown(f"Follow for daily updates: ({twit})")
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
        
        st.write("### Overview Metrics")
        aggregates = calculate_aggregates(df_filtered)
        st.table(pd.DataFrame(aggregates, index=[0]))
        df_transaction_counts = calculate_transaction_count(df_filtered)

        transactions_by_day = create_barchart(df_transaction_counts)
        st.write("### Transaction Count by Day")
        st.bar_chart(df_transaction_counts, height=400)
        
        st.write("### Initiation Rites vs. Completed Tasks by Day")
        task_data = calculate_tasks(df_filtered)
            
        st.line_chart(data=task_data, height=400)
        # Generate the leaderboard table
        leaderboard = calculate_leaderboard(df_filtered, from_address)
        st.write("### Leaderboard Taskbot + Corbanu")
        st.table(leaderboard)

        amount_by_day = calculate_amount_by_day(df_filtered, from_address)
        st.write("### Daily Earned PFT (Taskbot + Corbanu)")
        st.line_chart(data=amount_by_day, height=400)
        highlight_dict = {

    "r4yc85M1hwsegVGZ1pawpZPwj65SVs8PzD": {"color": "green",  "label": "task_node","size_override": 200},
    "rnQUEEg8yyjrwk9FhyXpKavHyCRJM9BDMW": {"color": "grey",  "label": "","size_override": 200},
    "rJ1mBMhEBKack5uTQvM8vWoAntbufyG9Yn": {"color": "orange",  "label": "ODV_node","size_override": 200},
    "rJUYompGetiVrHvKsNxR6HTVjYtX71mjfA": {"color": "blue",  "label": "expert_node","size_override": 200},
    "rnQUEEg8yyjrwk9FhyXpKavHyCRJM9BDMW": {"color": "grey",  "label": "","size_override": 200},
    "rpb7dex8DMLRXunDcTbbQeteCCYcyo9uSd": {"color": "pink",  "label": "church_node","size_override": 200},
    "rMEQBmJZ8e6fFGsPpqbhGNC3v4JvptojA4": {"color": "indigo",  "label": "image_node","size_override": 200},
    "rawQ8tEPb7EvAgQHDMRQGVEuq8r4QnjY8C": {"color": "yellow",  "label": "NFT_node","size_override": 200},
    "rKZDcpzRE5hxPUvTQ9S3y2aLBUUTECr1vN": {"color": "red",  "label": "AGTI_node","size_override": 200},        
            
}
        st.write("### Network Graph of Address Relationships")

        G = create_graph(df_filtered, highlight_dict=highlight_dict)

        # Plot the graph
        plot_graph(G, 
           min_node_size=22, max_node_size=250,
           min_edge_size=0.05,   max_edge_size=5,
           spacing_factor=800)

except FileNotFoundError:
    st.error("The file 'transactions.csv' could not be found or generated.")
