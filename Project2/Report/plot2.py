import matplotlib.pyplot as plt
import pandas as pd

HMM = [0.862, 0.785, 0.379]
HMML = [0.887, 0.833, 0.409]
CRF = [0.986, 0.794, 0.476]
CRFALL = [0.992, 0.873, 0.679]
CRFLARGE = [0.994, 0.849, 0.517]

list1 = [HMM, HMML, CRF, CRFALL, CRFLARGE]

gap = ["HMM", "HMM_large", "CRF", "CRF_1-5", "CRF_large"]

name = ["Training Acc", "Test Acc", "OOV Acc"]

df = pd.DataFrame(list1, gap, name)
#df = pd.DataFrame(list2, gap2, name2)
df.plot()
plt.show()