import pandas as pd, math
df = pd.read_csv('espoo_predict.csv')
n = math.ceil(len(df) / 3)
for i in range(n):
    df.iloc[i*3:(i+1)*3].to_csv(f'espoo_batch_{i+1}.csv', index=False)
print(f"Created {n} batch files")
