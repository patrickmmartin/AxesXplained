from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

datasets = [{"x":[1,2,3], "y":[1,4,9], "z":[0,0,0], "colour": "red"} for _ in range(6)]

for dataset in datasets:
    ax.plot(dataset["x"], dataset["y"], dataset["z"], color=dataset["colour"])

plt.show()
