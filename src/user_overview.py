import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def inspect_data(data):
    print("Data Info:")
    print(data.info())
    
    print("\nFirst Few Rows:")
    print(data.head())
    print("\nMissing Values:")
    
    print(data.isnull().sum())
    print("\nDescriptive Statistics:")
    
    print(data.describe())
def top_10_handsets(data):
    top_handsets = data['Handset Type'].value_counts().head(10)
    print("\nTop 10 Handsets:")
    print(top_handsets)

def top_3_manufacturers(data):
    top_manufacturers = data['Handset Manufacturer'].value_counts().head(3)
    print("\nTop 3 Manufacturers:")
    print(top_manufacturers)
    return top_manufacturers.index
def top_5_handsets_per_manufacturer(data, top_manufacturers):
    """
    Finds the top 5 handsets for each of the top 3 manufacturers.
    Args:
        data (pd.DataFrame): The dataset.
        top_manufacturers (pd.Index): Names of the top 3 manufacturers.
    """
    for manufacturer in top_manufacturers:
        print(f"\nTop 5 Handsets for {manufacturer}:")
        handsets = data[data['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
        print(handsets)

def aggregate_application_data(data):
    """
    Aggregates data for different application categories (e.g., Social Media, Google, etc.).
    Args:
        data (pd.DataFrame): The dataset.
    Returns:
        pd.DataFrame: Aggregated application data.
    """
  
    application_columns = {
        'Social Media': ['Social Media DL (Bytes)', 'Social Media UL (Bytes)'],
        'Google': ['Google DL (Bytes)', 'Google UL (Bytes)'],
        'Youtube': ['Youtube DL (Bytes)', 'Youtube UL (Bytes)'],
        'Netflix': ['Netflix DL (Bytes)', 'Netflix UL (Bytes)'],
        'Gaming': ['Gaming DL (Bytes)', 'Gaming UL (Bytes)'],
        'Other': ['Other DL (Bytes)', 'Other UL (Bytes)'],
    }

    aggregated_data = {}
    for app, cols in application_columns.items():
        dl_col, ul_col = cols
        total_dl = data[dl_col].sum()
        total_ul = data[ul_col].sum()
        total_data = total_dl + total_ul
        aggregated_data[app] = {'Total DL (Bytes)': total_dl, 'Total UL (Bytes)': total_ul, 'Total Data Volume': total_data}

    aggregated_df = pd.DataFrame.from_dict(aggregated_data, orient='index')
    aggregated_df = aggregated_df.sort_values(by='Total Data Volume', ascending=False)

    print("\nAggregated Application Data:")
    print(aggregated_df)
    return aggregated_df




def segment_users_by_session_duration(data):
    """
    Segments users into deciles based on total session duration.
    Args:
        data (pd.DataFrame): The dataset.
    Returns:
        pd.DataFrame: Dataset with decile information.
    """
  
    session_duration_col = 'Dur. (ms)' 

    if session_duration_col not in data.columns:
        print(f"Column '{session_duration_col}' not found in the dataset.")
        return data

  
    user_id_col = 'MSISDN/Number'  
    if user_id_col not in data.columns:
        print(f"Column '{user_id_col}' not found in the dataset.")
        return data

   
    total_duration_per_user = data.groupby(user_id_col)[session_duration_col].sum().reset_index()
    total_duration_per_user.columns = [user_id_col, 'Total Session Duration']

   
    total_duration_per_user['Decile'] = pd.qcut(total_duration_per_user['Total Session Duration'], 10, labels=False)

    print("\nUser Decile Segmentation Summary:")
    decile_summary = total_duration_per_user.groupby('Decile').agg({'Total Session Duration': ['mean', 'sum']})
    print(decile_summary)


    data = data.merge(total_duration_per_user[[user_id_col, 'Decile']], on=user_id_col, how='left')
    return data
def univariate_analysis(data, columns):
    """
    Performs univariate analysis on specified columns and visualizes distributions.
    Args:
        data (pd.DataFrame): The dataset.
        columns (list): List of columns to analyze.
    """
    for col in columns:
        if col in data.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(data[col], kde=True, bins=30, color='blue')
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()
def bivariate_analysis(data, x_col, y_col):
    """
    Performs bivariate analysis between two columns using scatter plots and correlation.
    Args:
        data (pd.DataFrame): The dataset.
        x_col (str): Column for the x-axis.
        y_col (str): Column for the y-axis.
    """
    if x_col in data.columns and y_col in data.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=data[x_col], y=data[y_col], alpha=0.7, color='purple')
        plt.title(f"{x_col} vs. {y_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

        
        corr = data[[x_col, y_col]].corr().iloc[0, 1]
        print(f"Correlation between {x_col} and {y_col}: {corr:.2f}")
def eda_insights(data):
    """
    Summarizes key findings from the dataset.
    Args:
        data (pd.DataFrame): The dataset.
    """
    print("\nKey Insights:")
    print(f"Average session duration: {data['Dur. (ms)'].mean():.2f} ms")
    print(f"Total data volume: {data['Total DL (Bytes)'].sum() + data['Total UL (Bytes)'].sum():.2e} Bytes")
    print(f"Number of unique users: {data['MSISDN/Number'].nunique()}")

def main():
    
    file_path = '../data/Copy of Week2_challenge_data_source(CSV).csv'
    data = load_data(file_path)
    inspect_data(data)
    top_10_handsets(data)
    top_manufacturers = top_3_manufacturers(data)
    top_5_handsets_per_manufacturer(data, top_manufacturers)
    aggregated_app_data = aggregate_application_data(data)
    data = segment_users_by_session_duration(data)
    univariate_columns = ['Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)']
    univariate_analysis(data, univariate_columns)
    bivariate_analysis(data, 'Dur. (ms)', 'Total DL (Bytes)')
    bivariate_analysis(data, 'Total DL (Bytes)', 'Total UL (Bytes)')
    eda_insights(data)
   

if __name__ == "__main__":
    main()
       
  

    
