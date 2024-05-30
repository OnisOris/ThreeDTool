import numpy as np


class Points:
    def __init__(self, xyz: np.ndarray[:, 3], color='green', s=1, marker='o', method='scatter', text=False):
        self.text = text
        self.method = method
        self.xyz = np.array(xyz)
        self.color = color
        self.s = s
        self.marker = marker

    def show(self, ax):
        if self.method == 'plot':
            ax.plot(self.xyz.T[0], self.xyz.T[1], self.xyz.T[2], color=self.color)
        elif self.method == 'scatter':
            ax.scatter(self.xyz.T[0], self.xyz.T[1], self.xyz.T[2], color=self.color, s=self.s, marker=self.marker)
        if self.text:
            for point in self.xyz:
                ax.text(point[0], point[1], point[2], f"center point: \n {point[0], point[1], point[2]}", color='blue')
