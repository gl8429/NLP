import matplotlib.pyplot as plt
import pandas as pd

Traing = [0.89, 0.999,0.999,0.999,0.999]
Test = [0.861, 0.928, 0.932, 0.938, 0.936]
OOV = [0.191, 0.253, 0.311, 0.328, 0.433]

HMM = [0.890, 0.861, 0.191]
CRF = [0.999, 0.926, 0.282]
CRFAB = [0.999, 0.932, 0.311]
CRFDE = [0.999, 0.938, 0.328]
CRFALL = [0.999, 0.936, 0.433]

list = [HMM, CRF, CRFAB, CRFDE, CRFALL]
gap = ["HMM", "CRF", "CRF_1,2", "CRF_4,5", "CRF_1-5"]
name = ["Training Acc", "Test Acc", "OOV Acc"]

df = pd.DataFrame(list, gap, name)
plt.figure()
df.plot()
plt.show()
