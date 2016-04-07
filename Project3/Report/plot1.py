import matplotlib.pyplot as plt
import pandas as pd

size1 = [0.717, 0.670, 0.686]
size2 = [0.766, 0.700, 0.713]
size3 = [0.779, 0.711, 0.724]
size4 = [0.790, 0.726, 0.730]
size5 = [0.797, 0.728, 0.743]
size7 = [0.806, 0.738, 0.748]
size10 = [0.811, 0.756, 0.760]
size13 = [0.820, 0.762, 0.772]
size16 = [0.824, 0.769, 0.780]
size20 = [0.829, 0.771, 0.783]
size25 = [0.831, 0.776, 0.785]
size30 = [0.835, 0.778, 0.785]
size35 = [0.835, 0.783, 0.790]

list = [size1, size2, size3, size4, size5, size7, size10, size13, size16, size20, size25, size30, size35]
gap = ["1000", "2000", "3000", "4000", "5000", "7000", "10000", "13000", "16000", "20000", "25000", "30000", "35000"]
name = ["WSJ_NO_WSJ", "WSJ_NO_Brown", "WSJ_Brown_Brown"]

df = pd.DataFrame(list, gap, name)
plt.figure()
df.plot()
plt.show()
