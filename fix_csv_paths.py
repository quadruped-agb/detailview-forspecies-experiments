import pandas as pd, os

df = pd.read_csv('/home/hafsa/DetailView/dev_sample_predict_input.csv')
df['filename'] = df['filename'].apply(lambda x: os.path.basename(x).replace('.las', '.laz'))
df.to_csv('/home/hafsa/DetailView/dev_sample_predict_input.csv', index=False)
print(df.head())
