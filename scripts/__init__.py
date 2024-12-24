import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pandas as pd
import os
from sqlalchemy import create_engine

def load_data(file_path, db_params):
    """
    Load data from a file into a PostgreSQL database using SQLAlchemy.

    Parameters:
    - file_path: str, path to the file (CSV, Excel, or SQL)
    - db_params: dict, database connection parameters

    Returns:
    - DataFrame: Loaded data as a pandas DataFrame
    """
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.csv':
        # Load CSV file
        df = pd.read_csv(file_path)
    elif file_extension == '.xlsx':
        # Load Excel file
        df = pd.read_excel(file_path)
    elif file_extension == '.sql':
        try:
            with open(file_path, 'r') as file:
                sql_script = file.read()

            # Create SQLAlchemy engine
            engine = create_engine(f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}")
            
            # Execute the SQL script
            with engine.connect() as connection:
                connection.execute(sql_script)

            # Optionally, fetch data from a specific table
            df = pd.read_sql_table('your_table_name', engine)

        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    else:
        raise ValueError("Unsupported file type: {}".format(file_extension))

    return df

# Define your database connection parameters
db_params = {
    'dbname': 'telecom',
    'user': 'postgres',
    'password': '30251421',
    'host': 'localhost',  
    'port': '5432'  
}

# Call the function with the SQL file and database parameters
data = load_data("C:\\Users\\nadew\\Downloads\\Data-20241222T074727Z-001\\Data\\telecom.sql", db_params)


load_data("C:\\Users\\nadew\\Downloads\\Data-20241222T074727Z-001\\Data\\telecom.sql")
# # Task 1.1: Aggregate per user
# df["Total_Data"] = df["DL"] + df["UL"]
# agg_df = df.groupby("Bearer Id").agg(
#     num_sessions=("Session_ID", "count"),
#     total_duration=("Session_Duration", "sum"),
#     total_download=("Download_Data", "sum"),
#     total_upload=("Upload_Data", "sum"),
#     total_data=("Total_Data", "sum")
# ).reset_index()

# # Task 1.2: Exploratory Data Analysis (EDA)
# # 1. Handle Missing Values
# agg_df.fillna(agg_df.mean(), inplace=True)

# # 2. Describe Variables and Data Types
# print(agg_df.info())
# print(agg_df.describe())

# # 3. Variable Transformation: Segment into decile classes
# agg_df["decile"] = pd.qcut(agg_df["total_duration"], 5, labels=[1, 2, 3, 4, 5])
# decile_summary = agg_df.groupby("decile").agg(total_data=("total_data", "sum"))
# print(decile_summary)

# # 4. Basic Metrics (Mean, Median, etc.)
# metrics = agg_df[["total_duration", "total_data", "total_download", "total_upload"]].agg(["mean", "median", "std"])
# print(metrics)

# # 5. Non-Graphical Univariate Analysis (Dispersion Parameters)
# dispersion = agg_df[["total_duration", "total_data"]].apply(lambda x: (x.max() - x.min()) / x.mean())
# print(f"Dispersion Parameters:\n{dispersion}")

# # 6. Graphical Univariate Analysis
# sns.histplot(agg_df["total_duration"], kde=True)
# plt.title("Total Duration Distribution")
# plt.show()

# sns.boxplot(data=agg_df[["total_download", "total_upload"]])
# plt.title("Download vs Upload Data")
# plt.show()

# # 7. Bivariate Analysis
# sns.scatterplot(x="total_download", y="total_upload", data=agg_df)
# plt.title("Download vs Upload Relationship")
# plt.show()

# # 8. Correlation Analysis
# correlation_matrix = agg_df[["total_download", "total_upload", "total_duration", "total_data"]].corr()
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
# plt.title("Correlation Matrix")
# plt.show()

# # 9. Dimensionality Reduction (PCA)
# features = ["total_download", "total_upload", "total_duration", "total_data"]
# X = agg_df[features]
# X_scaled = StandardScaler().fit_transform(X)

# pca = PCA(n_components=2)
# principal_components = pca.fit_transform(X_scaled)

# pca_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
# explained_variance = pca.explained_variance_ratio_
# print(f"Explained Variance Ratio: {explained_variance}")

# sns.scatterplot(x="PC1", y="PC2", data=pca_df)
# plt.title("PCA - Dimensionality Reduction")
# plt.show()

# # PCA Interpretation:
# # 1. PC1 accounts for the majority variance in total data and session duration.
# # 2. PC2 highlights variance in download and upload data.
# # 3. Most user behavior patterns are clustered tightly, indicating similarity.
# # 4. Outliers suggest unique usage behavior needing further investigation.
