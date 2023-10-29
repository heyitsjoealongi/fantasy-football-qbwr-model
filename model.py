import pandas as pd

# Import Historical Sample Data CSV
df = pd.read_csv("historical-sample-data.csv")
print(df)

# Create Pass Index
hist_data_samp = df
hist_data_samp = hist_data_samp[["Pass", "Outcome"]].set_index("Pass")
index_range = hist_data_samp.index.min(), hist_data_samp.index.max()
print(index_range)