import matplotlib.pyplot as plt
import pandas as pd


HMM = [0.863, 0.785, 0.383]
HMML = [0.889, 0.833, 0.409]
CRF = [0.996, 0.811, 0.479]
CRFALL = [0.993, 0.879, 0.658]
CRFLARGE = [0.996, 0.854, 0.519]

list1 = [HMM, HMML, CRF, CRFALL, CRFLARGE]

gap = ["HMM", "HMM_large", "CRF", "CRF_1-5", "CRF_large"]

name = ["Training Acc", "Test Acc", "OOV Acc"]

df = pd.DataFrame(list1, gap, name)
#df = pd.DataFrame(list2, gap2, name2)
df.plot()
plt.show()