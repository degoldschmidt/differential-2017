from pytrack_analysis.array import rle
import numpy as np
import pandas as pd

np.random.seed(42)
d = np.random.random_integers(1, 6, 100)
out = np.zeros(1000, dtype=np.int32)
counter = 0
for i, each in enumerate(out):
    out[i] = d[counter]
    if np.random.rand() < 0.05:
        counter += 1

print(out)

lens, pos, states = rle(out, dt=np.ones(out.shape)+1)


df = pd.DataFrame({})
df['states'] = states
df['pos'] = pos
df['lens'] = lens
print(df)
