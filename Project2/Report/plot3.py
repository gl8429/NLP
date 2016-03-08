import matplotlib.pyplot as plt
import pandas as pd

HMM = [0.902, 0.875, 0.231]
CRF = [0.998, 0.928, 0.293]
CRFAB = [0.999, 0.934, 0.241]
CRFDE = [0.999, 0.931, 0.367]
CRFALL = [0.999, 0.943, 0.449]

list = [HMM, CRF, CRFAB, CRFDE, CRFALL]
gap = ["HMM", "CRF", "CRF_1,2", "CRF_4,5", "CRF_1-5"]
name = ["Training Acc", "Test Acc", "OOV Acc"]

df = pd.DataFrame(list, gap, name)
plt.figure()
df.plot()
plt.show()
