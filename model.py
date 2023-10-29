import pandas as pd
import matplotlib.pyplot as plt

# Import Historical Sample Data CSV
df = pd.read_csv("historical-sample-data.csv")
print(df)

# Create Pass Index
hist_data_samp = df
hist_data_samp = hist_data_samp[["Pass", "Outcome"]].set_index("Pass")
index_range = hist_data_samp.index.min(), hist_data_samp.index.max()
print(index_range)

# Graph Historical Sample Data
hist_data_samp.plot()
plt.ylabel("Passes Caught")
_ = plt.title("Fantasy Football QB/WR - Historical Data")
plt.show()

