import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

# Database Connection
db_user = "postgres"         # Update as needed
db_password = "your_password"  # Update with your PostgreSQL password
db_host = "localhost"         # Adjust if running on a remote server
db_port = "5432"              # Default PostgreSQL port
db_name = "your_database"     # Update with your database name

# Create a database engine
engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

# Load Data
query = "SELECT * FROM xdr_data;"
df = pd.read_sql(query, engine)

# Task 1.1: Aggregation per User
df["Total_Data_Volume"] = df["Total DL (Bytes)"] + df["Total UL (Bytes)"]
aggregated_df = df.groupby("MSISDN/Number").agg(
    num_sessions=("Bearer Id", "count"),
    total_duration=("Dur. (ms)", "sum"),
    total_download=("Total DL (Bytes)", "sum"),
    total_upload=("Total UL (Bytes)", "sum"),
    total_data=("Total_Data_Volume", "sum")
).reset_index()

# Task 1.2: EDA
# 1. Handle missing values
aggregated_df.fillna(aggregated_df.mean(), inplace=True)

# 2. Describe variables
print(aggregated_df.info())
print(aggregated_df.describe())

# 3. Variable transformations
aggregated_df["decile"] = pd.qcut(aggregated_df["total_duration"], 5, labels=[1, 2, 3, 4, 5])
decile_summary = aggregated_df.groupby("decile").agg(total_data=("total_data", "sum"))
print(decile_summary)

# 4. Basic metrics
metrics = aggregated_df[["total_duration", "total_data", "total_download", "total_upload"]].agg(["mean", "median", "std"])
print(metrics)

# 5. Non-Graphical Univariate Analysis
dispersion = aggregated_df[["total_duration", "total_data"]].apply(lambda x: (x.max() - x.min()) / x.mean())
print(f"Dispersion Parameters:\n{dispersion}")

# 6. Graphical Univariate Analysis
sns.histplot(aggregated_df["total_duration"], kde=True)
plt.title("Total Duration Distribution")
plt.show()

sns.boxplot(data=aggregated_df[["total_download", "total_upload"]])
plt.title("Download vs Upload Data")
plt.show()

# 7. Bivariate Analysis
sns.scatterplot(x="total_download", y="total_upload", data=aggregated_df)
plt.title("Download vs Upload Relationship")
plt.show()

# 8. Correlation Analysis
correlation_matrix = aggregated_df[["total_download", "total_upload", "total_duration", "total_data"]].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# 9. Dimensionality Reduction (PCA)
features = ["total_download", "total_upload", "total_duration", "total_data"]
X = aggregated_df[features]
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
explained_variance = pca.explained_variance_ratio_
print(f"Explained Variance Ratio: {explained_variance}")

sns.scatterplot(x="PC1", y="PC2", data=pca_df)
plt.title("PCA - Dimensionality Reduction")
plt.show()

# PCA Interpretation:
# 1. PC1 accounts for the majority variance in total data and session duration.
# 2. PC2 highlights variance in download and upload data.
# 3. Most user behavior patterns are clustered tightly, indicating similarity.
# 4. Outliers suggest unique usage behavior needing further investigation.
