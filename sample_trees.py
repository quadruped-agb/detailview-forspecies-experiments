import pandas as pd

df = pd.read_csv('/home/hafsa/DetailView/tree_metadata_dev.csv')

n_species = df['species'].nunique()
per_species = max(1, 100 // n_species)

sample = (
    df.groupby('species', group_keys=False)[df.columns]
    .apply(lambda x: x.sample(min(len(x), per_species), random_state=42))
    .reset_index(drop=True)
)

sample['species_id'] = -999
sample.to_csv('/home/hafsa/DetailView/dev_sample_groundtruth.csv', index=False)
sample[['filename', 'species_id', 'tree_H']].to_csv('/home/hafsa/DetailView/dev_sample_predict_input.csv', index=False)

print(f"Sampled {len(sample)} trees across {sample['species'].nunique()} species")
