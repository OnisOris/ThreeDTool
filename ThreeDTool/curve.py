from __future__ import annotations
import numpy as np

class Curve:
    """
    Класс представляет траекторию в 3D пространстве
    """

    def __init__(self, curve_array: np.ndarray = np.array([])):
        """
        :param curve_array: массив траекторий типа [[x1, y1, z1], ...., [xn, yn, zn]]
        :type curve_array: np.ndarray
        """
        self.curve_array = np.array(curve_array)

    def show(self, ax) -> None:
        """
        Функция строит траекторию
        :param ax: Объект ax из matplotlib
        :type ax: matplotlib.Axes.ax
        """
        ax.plot(self.curve_array.T[0], self.curve_array.T[1], self.curve_array.T[2])

    def __getitem__(self, item):
        return self.curve_array[item]

    def union(self, curve: Curve) -> None:
        """
        Функция объединяет траектории между собой
        :param curve: Объект того же класса Curve
        :type curve: Curve
        :return: None
        """
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
    Класс, содержащий в себе координаты двух дополнительных векторов. Одна составляющая вектора выглядит следующим
    образом [x, y, z, p_1x, p_1y, p_1z, p_2x, p_2y, p_2z]
    """

    def __init__(self, curve_array: np.ndarray = np.array([]), length=0.1):
        """
        :param curve_array: массив траекторий типа
        [[x11, y11, z11, p_11x, p_11y, p_11z, p_12x, p_12y, p_12z],
         ....,
         [xn1, yn1, zn1, p_n1x, p_n1y, p_n1z, p_n2x, p_n2y, p_n2z]]
         :type curve_array: np.ndarray
        """
        super().__init__(curve_array)
        self.length = length

    def show(self, ax) -> None:
        """
        Функция отображает траекторию
        :param ax:
        :type ax: matplotlib.axes.Axes
        :return: None
        """
        for i, item in enumerate(self.curve_array):
            ax.quiver(item[0], item[1], item[2], item[3], item[4], item[5],
                      length=self.length, color='r')
            ax.quiver(item[0], item[1], item[2], item[6], item[7], item[8],
                      length=self.length * 6, color='b')


class Curve5xs(Curve):
    """
    Данный класс хранит в себе объекты класса Curve5x. Каждый новый объект говорит о разрыве траектории.
    """

    def __init__(self, curve_array: np.ndarray = np.array([]), length: float = 1):
        """
        :param curve_array: массив с объектами класса Curve5x
        :type curve_array: np.ndarray or list
        :param length: Длина отображения векторов
        :type length: float
        """
        super().__init__(curve_array)
        self.length = length

    def show(self, ax) -> None:
        """
        Функция отображает траекторию
        :param ax:
        :type ax: matplotlib.axes.Axes
        :return: None
        """
        for i, item in enumerate(self.curve_array):
            ax.quiver(item[0], item[1], item[2], item[3], item[4], item[5],
                      length=self.length, color='r')
            ax.quiver(item[0], item[1], item[2], item[6], item[7], item[8],
                      length=self.length * 10, color='b')
