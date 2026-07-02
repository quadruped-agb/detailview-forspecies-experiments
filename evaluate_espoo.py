import pandas as pd, glob, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SPECIES_MAP = {'Pine': 'Pinus_sylvestris', 'Spruce': 'Picea_abies',
               'Birch': 'Betula_pendula', 'Lime': 'Tilia_cordata'}

dfs = [pd.read_csv(f) for f in sorted(glob.glob('espoo_output/*/predictions_*.csv')) if 'probs' not in f]
pred = pd.concat(dfs, ignore_index=True)
pred['treeID'] = pred['filename'].str.replace('.laz', '', regex=False)

# keep only the latest prediction per tree (deduplicates rerun batches)
pred = pred.drop_duplicates(subset='treeID', keep='last')
print(f"Unique trees predicted: {len(pred)}")

gt = pd.read_csv('espoonlahti_groundtruth.csv')
gt['treeID'] = gt['treeID'].astype(str)
gt['true_species'] = gt['species'].map(SPECIES_MAP)

merged = pred.merge(gt[['treeID', 'true_species']], on='treeID', how='inner')
print(f"Matched {len(merged)} trees")

acc = accuracy_score(merged['true_species'], merged['species'])
report = classification_report(merged['true_species'], merged['species'], zero_division=0)
print(f"Accuracy: {acc:.2%}")
print(report)

with open('espoo_evaluation.txt', 'w') as f:
    f.write(f"Matched {len(merged)} trees\nAccuracy: {acc:.2%}\n\n{report}")

labels = sorted(merged['true_species'].unique())
cm = confusion_matrix(merged['true_species'], merged['species'], labels=labels)
n = len(labels)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticklabels(labels)
thresh = cm.max() / 2.0
for i in range(n):
    for j in range(n):
        if cm[i,j] > 0:
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color='white' if cm[i,j] > thresh else 'black')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'DetailView Espoonlahti — {acc:.2%} ({len(merged)} trees)')
plt.tight_layout()
plt.savefig('espoo_confusion_matrix.png', dpi=150)
print("Saved espoo_confusion_matrix.png")
