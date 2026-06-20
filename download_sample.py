import pandas as pd
import requests
import os

df = pd.read_csv('/home/hafsa/tree_metadata_dev.csv')
# pick 10 trees from different species
sample = df.groupby('species').first().head(10).reset_index()

os.makedirs('sample_trees', exist_ok=True)
for _, row in sample.iterrows():
    url = f"https://zenodo.org/records/13255198/files/{row['treeID']}.laz?download=1"
    r = requests.get(url)
    with open(f"sample_trees/{row['treeID']}.laz", 'wb') as f:
        f.write(r.content)
    print(f"Downloaded {row['treeID']}")

