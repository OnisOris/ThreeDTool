import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('TkAgg')

class Dspl:
    def __init__(self, input_array, qt=True):
        if qt:
            mpl.use('Qt5Agg')
        self.input_array = input_array
        self.fig = None
        self.ax = None
        self.create_subplot3D()
        # input_array[0].show(self.ax)

    def create_subplot3D(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Y Label')
        self.ax.set_zlabel('Z Label')
    def show(self):
        for obj in self.input_array:
            obj.show(self.ax)
        plt.show()



