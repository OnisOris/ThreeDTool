import numpy as np
class Points:
    """
    Класс, хранящий в себе точки вида [x, y, z]
    """
    def __init__(self, xyz: np.ndarray,
                 signs: list[str] | np.ndarray[list] = None,
                 color: str = 'green',
                 s: int = 1,
                 marker: str = 'o',
                 method: str = 'scatter',
                 text: bool = False):
        """
        :param xyz: Массив точек nx3
        :type xyz: np.ndarray
        :param color: Цвет точек или траектории
        :type color: str
        :param s: int
        :param marker: Маркер точек
        :type marker: str
        :param method: Показ точек точками или траекториями
        :type method: str
        :param text: True - подписать точки
        :type text: bool
        """
        self.signs = signs
        if signs is not None:
            if np.shape(signs)[0] != np.shape(xyz)[0]:
                raise Exception("Size 0 axis of signs and xyz must be the same")
        self.text = text
        self.method = method
        self.xyz = np.array(xyz)
        self.color = color
        self.s = s
        self.marker = marker

    def show(self, ax) -> None:
        """
        Функция для отображения точек
        :param ax: Объект matplotlib.axes.Axes
        :type ax: matplotlib.axes.Axes
        :return: None
        """
        if self.method == 'plot':
            ax.plot(self.xyz.T[0], self.xyz.T[1], self.xyz.T[2], color=self.color)
        elif self.method == 'scatter':
            ax.scatter(self.xyz.T[0], self.xyz.T[1], self.xyz.T[2], color=self.color, s=self.s, marker=self.marker)
        if self.text:
            if self.signs != None:
                for i, point in enumerate(self.xyz):
                    ax.text(point[0], point[1], point[2],
                            f"{sign[i]}\n{point[0], point[1], point[2]}", color='blue')
            else:
                for i, point in enumerate(self.xyz):
                    ax.text(point[0], point[1], point[2],
                            f"{point[0], point[1], point[2]}", color='blue')

