import pandas as pd
import numpy as np

# List of CSV file paths
csv_files = [r"/kaggle/working/tb_predictions1", r"/kaggle/working/tb_predictions2", r"/kaggle/working/tb_predictions3"]

# Read all CSV files and store the 'Target' columns
dfs = [pd.read_csv(file) for file in csv_files]

# Ensure all DataFrames have the same order
for df in dfs:
    df.sort_values(by=df.columns[0], inplace=True)
    df.reset_index(drop=True, inplace=True)

# Stack 'Target' columns and compute the average
target_avg = np.mean([df['Target'] for df in dfs], axis=0)

# Convert to binary output (1 if avg >= 0.5, else 0)
final_target = (target_avg >= 0.5).astype(int)

# Create final submission DataFrame
final_submission = dfs[0].copy()
final_submission['Target'] = final_target

# Save final submission CSV
final_submission.to_csv("final_submission.csv", index=False)

print("Final submission file saved as final_submission.csv")
