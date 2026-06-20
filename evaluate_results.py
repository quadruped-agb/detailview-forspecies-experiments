import pandas as pd
import glob
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load and combine all batch prediction files
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

# --- Confusion matrix ---
labels = sorted(merged['species_y'].unique())
cm = confusion_matrix(merged['species_y'], merged['species_x'], labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.to_csv('confusion_matrix.csv')

n = len(labels)
fig, ax = plt.subplots(figsize=(max(14, n * 0.4), max(12, n * 0.4)))
im = ax.imshow(cm, cmap='Blues')

ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(labels, rotation=90, fontsize=7)
ax.set_yticklabels(labels, fontsize=7)

# Gridlines between cells
ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
ax.tick_params(which='minor', bottom=False, left=False)

# Count labels inside cells
thresh = cm.max() / 2.0
for i in range(n):
    for j in range(n):
        if cm[i, j] > 0:
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=6, color='white' if cm[i, j] > thresh else 'black')

ax.set_xlabel('Predicted species')
ax.set_ylabel('True species')
ax.set_title(f'Confusion matrix — accuracy: {acc:.2%} ({len(merged)} trees)')
fig.colorbar(im, ax=ax, label='Count')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("Saved confusion_matrix.csv and confusion_matrix.png")
