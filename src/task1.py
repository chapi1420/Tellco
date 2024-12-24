from sqlalchemy import create_engine
import pandas as pd

# Database connection details
db_user = "postgres"
db_password = "30251421"  
db_host = "localhost"
db_port = "5432"
db_name = "telecom"

# Create a database engine using pg8000
engine = create_engine(f"postgresql+pg8000://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

# Query the data
query = "SELECT * FROM xdr_data;"
df = pd.read_sql(query, engine)

# Aggregate the xDR data
df["Total_Data_Volume"] = df["Total DL (Bytes)"] + df["Total UL (Bytes)"]
aggregated_df = df.groupby("MSISDN/Number").agg(
    num_sessions=("Bearer Id", "count"),
    total_duration=("Dur. (ms)", "sum"),
    total_download=("Total DL (Bytes)", "sum"),
    total_upload=("Total UL (Bytes)", "sum"),
    total_data=("Total_Data_Volume", "sum")
).reset_index()

print(aggregated_df.head())
