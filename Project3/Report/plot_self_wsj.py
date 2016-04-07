import matplotlib.pyplot as plt
import pandas as pd

size1 = [0.742]
size2 = [0.747]
size3 = [0.750]
size4 = [0.752]
size5 = [0.749]
size7 = [0.753]
size10 = [0.759]
size13 = [0.762]
size17 = [0.761]
size21 = [0.762]

list = [size1, size2, size3, size4, size5, size7, size10, size13, size17, size21]
gap = ["1000", "2000", "3000", "4000", "5000", "7000", "10000", "13000", "17000", "21000"]
name = ["Self Training Brown"]

df = pd.DataFrame(list, gap, name)
plt.figure()
df.plot()
plt.show()
