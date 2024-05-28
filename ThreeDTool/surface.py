import numpy as np
from loguru import logger

class Sphere:
    """
    Класс сферы. Основано на уравнении (x-a)^2 + (y - b)^2 + (z - c)^2 = r^2
    """

    def __init__(self, a=0, b=0, c=0, r=1):
        self.r = r
        self.a = a
        self.b = b
        self.c = c

    def point_analyze(self, point: np.ndarray or list):
        """
        Функция анализирует, находится ли входная точка внутри сферы
        :param point:
        :return:
        """
        logger.debug(point)
        eq = (point[0] - self.a) ** 2 + (point[1] - self.b) ** 2 + (point[2] - self.c) ** 2
        logger.debug(eq)
        if eq < self.r ** 2:
            return True
        else:
            return False

    def array_analyze(self, array: np.ndarray):
        """
        Функция принимает в себя массив nx3 и убирает все точки в нем, которые не входят внутрь сферы
        :param array:
        :return:
        """
        arr_out = np.array([0, 0, 0])
        for item in array:
            if self.point_analyze(item):
                arr_out = np.vstack([arr_out, item])

        return arr_out


