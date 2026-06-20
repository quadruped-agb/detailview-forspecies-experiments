import pandas as pd, os

df = pd.read_csv('/home/hafsa/DetailView/dev_sample_groundtruth.csv')
with open('/home/hafsa/DetailView/extract_list.txt', 'w') as f:
    for fn in df['filename']:
        base = os.path.basename(fn).replace('.las', '.laz')
        f.write(base + '\n')

print(f"Wrote {len(df)} filenames to extract_list.txt")
