import numpy as np

class Sphere:
    """
    Класс сферы. Основано на уравнении (x-a)^2 + (y - b)^2 + (z - c)^2 = r^2
    """

    def __init__(self, a: float = 0,
                 b: float = 0,
                 c: float = 0,
                 r: float = 1):
        """
        :param a: Центр по x
        :type a: float
        :param b: Центр по y
        :type b: float
        :param c: Центр по z
        :type c: float
        :param r: Радиус сферы
        :type r: float
        """
        self.r = r
        self.a = a
        self.b = b
        self.c = c

    def point_analyze(self, point: np.ndarray | list) -> bool:
        """
        Функция анализирует, находится ли входная точка внутри сферы
        :param point: Точка типа [x, y, z]
        :type point: np.ndarray | list
        :return: bool
        """
        eq = (point[0] - self.a) ** 2 + (point[1] - self.b) ** 2 + (point[2] - self.c) ** 2
        if eq < self.r ** 2:
            return True
        else:
            return False

    def point_analyze_not_eq(self, point: np.ndarray | list) -> bool:
        """
        Функция анализирует, находится ли входная точка внутри сферы включительно границы
        :param point: Точка типа [x, y, z]
        :type point: np.ndarray | list
        :return: bool
        """
        eq = (point[0] - self.a) ** 2 + (point[1] - self.b) ** 2 + (point[2] - self.c) ** 2
        r = self.r ** 2
        if eq <= self.r ** 2:
            return True
        else:
            return False

    def array_analyze(self, array: np.ndarray) -> np.ndarray:
        """
        Функция принимает в себя массив nx3 и убирает все точки в нем, которые не входят внутрь сферы
        :param array: Массив точек размерностью nx3
        :type array: np.ndarray
        :return: np.ndarray
        """
        arr_out = np.array([0, 0, 0])
        for item in array:
            if self.point_analyze(item):
                arr_out = np.vstack([arr_out, item])

        return arr_out


