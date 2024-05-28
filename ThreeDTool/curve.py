import numpy as np
from loguru import logger


class Curve():
    """
    Класс представляет траекторию в 3D пространстве
    """

    def __init__(self, curve_array: np.ndarray = np.array([])):
        self.curve_array = np.array(curve_array)

    # @property
    # def curve_array(self):
    #     return self.curve_array
    #
    # @curve_array.setter
    # def __curve_array(self, input_array):
    #     self.__curve_array = input_array

    def show(self, ax):
        ax.plot(self.curve_array.T[0], self.curve_array.T[1], self.curve_array.T[2])

    def __getitem__(self, item):
        return self.curve_array[item]

    # def add(self, ):
    # def __add__(self, other):
    #     return

    def union(self, curve):
        if self.curve_array.shape[0] == 0:
            if isinstance(curve, Curve):
                self.curve_array = np.hstack((self.curve_array, curve.curve_array))
            else:
                self.curve_array = np.hstack((self.curve_array, curve))
        else:
            if isinstance(curve, Curve):
                self.curve_array = np.vstack((self.curve_array, curve.curve_array))
            else:
                self.curve_array = np.vstack((self.curve_array, curve))


class Curve5x(Curve):
    """
    """

    def __init__(self, curve_array: np.ndarray = np.array([]), length=0.1):
        self.curve_array = curve_array
        self.length = length

    def show(self, ax):
        for i, item in enumerate(self.curve_array):
            # logger.debug(f"{item[0]}, {item[1]}, {item[2]}, {item[0]} + {item[3]}, {item[1]} + {item[4]}, {item[2]} + {item[5]}")
            ax.quiver(item[0], item[1], item[2], item[3], item[4], item[5],
                      length=self.length, color='r')
            ax.quiver(item[0], item[1], item[2], item[6], item[7], item[8],
                      length=self.length*6, color='b')


class Curve5xs(Curve):
    """
    данный класс хранит в себе объекты класса Curve5x. Каждый новый объект говорит о разрыве траектории.
    """

    def __init__(self, curve_array: np.ndarray = np.array([]), length=1):
        self.curve_array = curve_array
        self.length = length

    def show(self, ax):
        for i, item in enumerate(self.curve_array):
            # logger.debug(f"{item[0]}, {item[1]}, {item[2]}, {item[0]} + {item[3]}, {item[1]} + {item[4]}, {item[2]} + {item[5]}")
            ax.quiver(item[0], item[1], item[2], item[3], item[4], item[5],
                      length=self.length, color='r')
            ax.quiver(item[0], item[1], item[2], item[6], item[7], item[8],
                      length=self.length * 10, color='b')
