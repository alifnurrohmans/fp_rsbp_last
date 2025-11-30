import pandas as pd
import numpy as np

np.random.seed(42)

rows = 1000
data = {}

# Generate jawaban kuisioner 1-5
for i in range(1, 21):
    data[f"q{i}"] = np.random.randint(1, 6, rows)

df = pd.DataFrame(data)

# Label multi-career berdasarkan pola jawaban
df["offensive"]   = ((df.q1+df.q2+df.q3+df.q4)   / 4 > 3.5).astype(int)
df["blue_team"]   = ((df.q5+df.q6+df.q7)         / 3 > 3.5).astype(int)
df["malware"]     = ((df.q8+df.q9)               / 2 > 3.7).astype(int)
df["forensics"]   = ((df.q10+df.q11)             / 2 > 3.5).astype(int)
df["network"]     = ((df.q12+df.q13)             / 2 > 3.5).astype(int)
df["cloud"]       = ((df.q14+df.q15)             / 2 > 3.5).astype(int)
df["appsec"]      = ((df.q16+df.q17)             / 2 > 3.5).astype(int)
df["threatintel"] = ((df.q18+df.q19)             / 2 > 3.7).astype(int)
df["grc"]         = (df.q20 > 4).astype(int)

df.to_csv("cyber_dataset.csv", index=False)
print("Dataset berhasil dibuat!")
