import pandas as pd

from data_exploration import explore_data
from statistics_analysis import descriptive_statistics
from probability_analysis import probability_analysis
from hypothesis_testing import run_ttest
from visualization import create_graphs
from regression_model import run_regression

# Load Dataset

df = pd.read_csv("dataset/psx_final_dataset.csv")

# Run Modules

explore_data(df)

descriptive_statistics(df)

probability_analysis(df)

run_ttest(df)

create_graphs(df)

run_regression(df)

print("\nCCP PROJECT COMPLETED")