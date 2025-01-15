import subprocess
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Run main.py if CSV does not exist
if not os.path.exists("xrp_token_payments.csv"):
    st.info("Generating transactions.csv using main.py...")
    subprocess.run(["python", "transaction_script.py"], check=True)

# Function to load the CSV file
@st.cache_data
def load_csv():
    return pd.read_csv("transactions.csv")

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

except FileNotFoundError:
    st.error("The file 'transactions.csv' could not be found or generated.")
