import pandas as pd

def explore_data(df):

    print("\n==========================")
    print("DATA EXPLORATION")
    print("==========================")

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nCompanies:")
    print(df["Symbol"].unique())

    print("\nRecords Per Company:")
    print(df["Symbol"].value_counts())