import matplotlib.pyplot as plt
import pandas as pd

size1 = [0.699, 0.644, 0.663]
size2 = [0.730, 0.683, 0.697]
size3 = [0.750, 0.700, 0.715]
size4 = [0.766, 0.710, 0.730]
size5 = [0.772, 0.718, 0.739]
size7 = [0.785, 0.731, 0.742]
size10 = [0.796, 0.730, 0.740]
size13 = [0.801, 0.736, 0.746]
size17 = [0.807, 0.743, 0.753]
size21 = [0.811, 0.745, 0.758]

list = [size1, size2, size3, size4, size5, size7, size10, size13, size17, size21]
gap = ["1000", "2000", "3000", "4000", "5000", "7000", "10000", "13000", "17000", "21000"]
name = ["Brown_NO_Brown", "Brown_NO_WSJ", "Brown_WSJ_WSJ"]

df = pd.DataFrame(list, gap, name)
plt.figure()
df.plot()
plt.show()
