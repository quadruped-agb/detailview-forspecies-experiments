import pandas as pd
import glob
from sklearn.metrics import accuracy_score, classification_report

dfs = []
for i in range(1, 20):
    for f in glob.glob(f'output/{i}/predictions_*.csv'):
        if 'probs' not in f:
            dfs.append(pd.read_csv(f))

pred = pd.concat(dfs, ignore_index=True)
pred['treeID'] = pred['filename'].str.replace('.laz', '', regex=False).str.lstrip('0').replace('', '0').astype(int)

gt = pd.read_csv('dev_sample_groundtruth.csv')
merged = pred.merge(gt[['treeID', 'species']], on='treeID', how='inner')

acc = accuracy_score(merged['species_y'], merged['species_x'])
report = classification_report(merged['species_y'], merged['species_x'], zero_division=0)

print(f"Matched {len(merged)} trees out of {len(pred)} predictions")
print(f"Accuracy: {acc:.2%}")
print(report)

with open('evaluation_results.txt', 'w') as f:
    f.write(f"Matched {len(merged)} trees out of {len(pred)} predictions\n")
    f.write(f"Accuracy: {acc:.2%}\n\n")
    f.write(report)

merged.to_csv('merged_results.csv', index=False)
