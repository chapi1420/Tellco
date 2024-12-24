import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
 #Database connection details
db_user = "postgres"
db_password = "30251421"  
db_host = "localhost"
db_port = "5432"
db_name = "telecom"
engine = create_engine(f"postgresql+pg8000://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
query = "SELECT * FROM public.xdr_data;"
data = pd.read_sql(query, engine)
def aggregate_customer_data(data):
    data['TCP DL Retrans. Vol (Bytes)'] = data['TCP DL Retrans. Vol (Bytes)'].replace(np.nan, data['TCP DL Retrans. Vol (Bytes)'].mean())
    data['Avg RTT DL (ms)'] = data['Avg RTT DL (ms)'].replace(np.nan, data['Avg RTT DL (ms)'].mean())
    data['Avg Bearer TP DL (kbps)'] = data['Avg Bearer TP DL (kbps)'].replace(np.nan, data['Avg Bearer TP DL (kbps)'].mean())
    
    aggregated_data = data.groupby('MSISDN/Number').agg({
        'TCP DL Retrans. Vol (Bytes)': 'mean',
        'Avg RTT DL (ms)': 'mean',
        'Handset Type': 'first',
        'Avg Bearer TP DL (kbps)': 'mean'
    }).reset_index()

    return aggregated_data

aggregated_data = aggregate_customer_data(data)
print(aggregated_data)
def compute_top_bottom_frequent(data):
    top_tcp = data['TCP DL Retrans. Vol (Bytes)'].nlargest(10)
    bottom_tcp = data['TCP DL Retrans. Vol (Bytes)'].nsmallest(10)
    most_frequent_tcp = data['TCP DL Retrans. Vol (Bytes)'].mode()

    top_rtt = data['Avg RTT DL (ms)'].nlargest(10)
    bottom_rtt = data['Avg RTT DL (ms)'].nsmallest(10)
    most_frequent_rtt = data['Avg RTT DL (ms)'].mode()

    top_throughput = data['Avg Bearer TP DL (kbps)'].nlargest(10)
    bottom_throughput = data['Avg Bearer TP DL (kbps)'].nsmallest(10)
    most_frequent_throughput = data['Avg Bearer TP DL (kbps)'].mode()

    return {
        'top_tcp': top_tcp,
        'bottom_tcp': bottom_tcp,
        'most_frequent_tcp': most_frequent_tcp,
        'top_rtt': top_rtt,
        'bottom_rtt': bottom_rtt,
        'most_frequent_rtt': most_frequent_rtt,
        'top_throughput': top_throughput,
        'bottom_throughput': bottom_throughput,
        'most_frequent_throughput': most_frequent_throughput
    }

top_bottom_frequent = compute_top_bottom_frequent(aggregated_data)
print(top_bottom_frequent)
def plot_throughput_distribution(data):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Handset Type', y='Avg Bearer TP DL (kbps)', data=data)
    plt.title('Distribution of Average Throughput per Handset Type')
    plt.xticks(rotation=45)
    plt.show()

plot_throughput_distribution(aggregated_data)