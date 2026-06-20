import pandas as pd
import glob

all_files = sorted(glob.glob('/home/hafsa/DetailView/output/*/predictions_*.csv'))
# exclude probability files (they have "probs" in filename)
pred_files = [f for f in all_files if 'probs' not in f]

print(f"Found {len(pred_files)} prediction files")

dfs = [pd.read_csv(f) for f in pred_files]
combined = pd.concat(dfs, ignore_index=True)
combined.to_csv('/home/hafsa/DetailView/all_predictions_clean.csv', index=False)
print(f"Combined {len(combined)} total predictions")
print(combined.head())
