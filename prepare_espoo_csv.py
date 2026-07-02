import pandas as pd
import os

gt = pd.read_csv('espoonlahti_groundtruth.csv')
gt['treeID'] = gt['treeID'].astype(str)

ref = pd.read_csv('MLS_species_ref.csv')
ref.columns = ref.columns.str.strip()
ref = ref.rename(columns={'id': 'treeID', 'height (m)': 'tree_H'})
ref['treeID'] = ref['treeID'].astype(str)

merged = gt.merge(ref[['treeID', 'tree_H']], on='treeID', how='left')
merged = merged.dropna(subset=['tree_H'])

merged['filename'] = merged['treeID'] + '.laz'
merged['species_id'] = -999
merged[['filename', 'species_id', 'tree_H']].to_csv('espoo_predict.csv', index=False)
print(f"Saved espoo_predict.csv with {len(merged)} trees")
print(merged['species'].value_counts())
