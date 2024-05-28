from math import sqrt
from typing import Any

from loguru import logger
import numpy as np
from numpy import ndarray, dtype


# from threeDTool import *


class Line:
    # Уравнение вида:
    # (x-a)/p1 = (y-b)/p2 = (z-c)/p3
    def __init__(self, a=0, b=0, c=0, p1=1, p2=0, p3=0, logging=False):
        self._a = a
        self._b = b
        self._c = c
        self._p1 = p1
        self._p2 = p2
        self._p3 = p3
        self.logging = logging

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c

    @property
    def p1(self):
        return self._p1

    @property
    def p2(self):
        return self._p2

    @property
    def p3(self):
        return self._p3

    @a.setter
    def a(self, a):
        self._a = a

    @b.setter
    def b(self, b):
        self._b = b

    @c.setter
    def c(self, c):
        self._c = c

    @p1.setter
    def p1(self, p1):
        self._p1 = p1

    @p2.setter
    def p2(self, p2):
        self._p2 = p2

    @p3.setter
    def p3(self, p3):
        self._p3 = p3

    def info(self) -> None:
        logger.debug(
            f'a = {self._a}, b = {self._b}, c = {self._c}, p1 = {self._p1}, p2 = {self._p2}, p3 = {self._p3}')

    def coeffs(self) -> ndarray:
        return np.array([self._a, self._b, self._c, self._p1, self._p2, self._p3])

    # Создание коэффициентов прямой по двум точкам в пространстве.
    # Принимает точку в виде массива 1x3 объекта класса numpy.array с тремя координатами [x, y, z]
    # point1 = [x1, y1, z1]
    def line_create_from_points(self, point1, point2) -> None:
        """
        Создает коэффициенты прямой по двум точкам в пространстве.
        Принимает точку в виде массива 1x3 объекта класса numpy.array с тремя координатами [x, y, z]
        :param point1: [x1, y1, z1]
        :param point2: [x2, y2, z2]
        :return: None
        """
        if np.shape(point1)[0] == 2:
            self._c = 0
            p3 = 0
        else:
            self._c = point1[2]
            p3 = point2[2] - point1[2]
        self._a = point1[0]
        self._b = point1[1]

        # TODO: проверка на нуль
        p1 = point2[0] - point1[0]
        p2 = point2[1] - point1[1]

        if p1 == 0 and p2 == 0 and p3 == 0:
            if self.logging:
                logger.error("Создать линию из двух одинаковых точек нельзя")
        else:
            mod_N = sqrt(p1 ** 2 + p2 ** 2 + p3 ** 2)
            # Проверка на равенство длины вектора нормали единице
            if mod_N != 1.0:
                p1 = p1 / mod_N
                p2 = p2 / mod_N
                p3 = p3 / mod_N
            self._p1 = p1
            self._p2 = p2
            self._p3 = p3

    def line_from_planes(self, plane1, plane2):
        '''
        Функция, создающая линию из двух пересекающихся плоскостей.
        :param plane1:
        :param plane2:
        :return: True, если mod_p - нулевой вектор. False, если плоскости не пересекаются,
        либо параллельны, либо совпадают
        '''

        # Векторное произведение векторов нормали n_1 b n_2
        p1 = plane1.b * plane2.c - plane2.b * plane1.c  # проверено
        p2 = plane1.c * plane2.a - plane2.c * plane1.a
        p3 = plane1.a * plane2.b - plane2.a * plane1.b
        mod_p = np.linalg.norm(np.cross(plane1.get_N(), plane2.get_N()))
        # Сначала проверяем не параллельны ли эти две плоскости:
        if mod_p != 0:
            if mod_p != 1.0 and mod_p != 0:
                p1 = p1 / mod_p
                p2 = p2 / mod_p
                p3 = p3 / mod_p
            elif mod_p == 0:
                logger.error("P - нулевой вектор")
                return True
            self._p1 = p1
            self._p2 = p2
            self._p3 = p3
            # как найти точку с прямой?
            # если прямая не параллельна x и y, мы можем занулить координату z (c = 0)
            # Но если она параллельна, то одна из плоскостей plane1 или plane 2 параллельна осям x и y, тогда необходимо
            # занулить другую координату, например x или y

            # проверяем параллельность плоскостей осям x и y

            # # Если не параллельны, то:
            # z = 0
            val1_1 = plane1.a * plane2.b - plane2.a * plane1.b
            val1_2 = plane2.a * plane1.b - plane1.a * plane2.b
            # y = 0
            val2_1 = plane2.c * plane1.a - plane1.c * plane2.a
            val2_2 = plane2.a * plane1.c - plane1.a * plane2.c
            # x = 0
            val3_1 = plane2.c * plane1.b - plane1.c * plane2.b
            val3_2 = plane2.c * plane1.b - plane1.c * plane2.b
            if val1_1 != 0 and plane2.b != 0:
                self._c = 0
                self._a = (plane2.d * plane1.b - plane1.d * plane2.b) / val1_1
                self._b = - (plane2.a * self._a + plane2.d) / plane2.b
            elif val1_2 != 0 and plane2.b == 0:
                self._c = 0
                self._b = (plane1.a * plane2.d - plane1.d * plane2.a) / val1_2
                self._a = - (plane2.b * self._c + plane2.d) / plane2.a

            elif val2_1 != 0 and plane2.c != 0:
                self._b = 0
                self._a = (plane2.d * plane1.c - plane1.d * plane2.c) / val2_1
                self._c = -(plane2.a * self._a + plane2.d) / plane2.c

            elif val2_2 != 0 and plane2.a != 0:
                self._b = 0
                self._c = (plane1.a * plane2.d - plane2.a * plane1.d) / val2_2
                self._a = -(plane2.c * self._c + plane2.d) / plane2.a

            elif val3_1 != 0 and plane2.c != 0:
                self._a = 0
                self._b = (plane1.c * plane2.d - plane1.d * plane2.c) / val3_1
                self._c = -(plane2.b * self._b + plane2.d) / plane2.c
            elif val3_2 != 0 and plane2.b != 0:
                self._a = 0
                self._c = (plane2.b * plane1.d - plane2.d * plane1.b) / val3_2
                self._b = -(plane2.c * self._c + plane2.d) / plane2.b
            else:
                logger.debug("Zero Error")
        else:
            logger.debug("Плоскости не пересекаются и либо параллельны, либо совпадают")
            return False

    def point_belongs_to_the_line(self, point):
        """
        Функция, определющая, принадлежит ли точка прямой
        :param point: список из координат [x, y, z]
        :return: True, если принадлежит, False, если не принадлежит
        """
        eq1 = np.round(self.p2 * self.p3 * (point[0] - self.a) - self.p1 * self.p3 * (point[1] - self.b), 8)
        eq2 = np.round(self.p1 * self.p3 * (point[1] - self.b) - self.p1 * self.p2 * (point[2] - self.c), 8)
        eq3 = np.round(self.p1 * self.p2 * (point[2] - self.c) - self.p2 * self.p3 * (point[0] - self.a), 8)
        if eq1 == 0 and eq2 == 0 and eq3 == 0:
            return True
        else:
            return False

    def equation_y(self):
        """
        Данная функция возвращает коэффициенты k_1, b, k_2, c из уравнения y = k_1*x + b + k_2*z + c
        :return:
        """
        return {"k_1": self.p2 / self.p1,
                "k_2": self.p2 / self.p3,
                "b": self.b - self.a * self.p2 / self.p1,
                "c": self.c * self.p2 / self.p3
                }

    def show(self, ax):
        p = self.coeffs()[3:6]
        abc = self.coeffs()[0:3]
        pabc = p * 15 + abc
        pmabc = -p * 15 + abc
        points = np.vstack([pabc, pmabc])
        x = points.T[0]
        y = points.T[1]
        z = points.T[2]
        ax.plot(x, y, z)



class Line_segment(Line):
    def __init__(self, a=0, b=0, c=0, p1=1, p2=0, p3=0, point1=None, point2=None):
        if point1 is not None and point2 is not None:
            super().__init__(a, b, c, p1, p2, p3)
            self.segment_create_from_points(point1, point2)
        else:
            point1 = np.array([0, 0, 0])
            point2 = np.array([1, 0, 0])
            super().__init__(a, b, c, p1, p2, p3)
        self.color = 'blue'
        self.linewidth = 2
        self.point1 = np.array(point1)
        self.point2 = np.array(point2)
        # Сортировка по возрастанию для удобства дальнейшей работы
        self.border_x = np.array([point1[0], point2[0]])
        self.border_y = np.array([point1[1], point2[1]])
        self.border_z = np.array([point1[2], point2[2]])
        self.border_x.sort()
        self.border_y.sort()
        self.border_z.sort()
        self.lenth = np.linalg.norm(self.point1 - self.point2)

    # @property
    # def a(self):
    #     return self.__a
    #
    # @property
    # def b(self):
    #     return self.__b
    #
    # @property
    # def c(self):
    #     return self.__c
    #
    # @property
    # def p1(self):
    #     return self.__p1
    #
    # @property
    # def p2(self):
    #     return self.__p2
    #
    # @property
    # def p3(self):
    #     return self.__p3
    #
    # @a.setter
    # def a(self, a):
    #     self.__a = a
    #
    # @b.setter
    # def b(self, b):
    #     self.__b = b
    #
    # @c.setter
    # def c(self, c):
    #     self.__c = c
    #
    # @p1.setter
    # def p1(self, p1):
    #     self.__p1 = p1
    #
    # @p2.setter
    # def p2(self, p2):
    #     self.__p2 = p2
    #
    # @p3.setter
    # def p3(self, p3):
    #     self.__p3 = p3

    def info(self) -> None:
        logger.debug(
            f'a = {self._a}, b = {self._b}, c = {self._c}, p1 = {self._p1}, p2 = {self._p2}, p3 = {self._p3}'
            f'\npoint1 = {self.point1}, point2 = {self.point2}')

    def coeffs(self) -> ndarray:
        return np.array([self._a, self._b, self._c, self._p1, self._p2, self._p3])

    def segment_create_from_points(self, point1, point2) -> None:
        """
        Создает коэффициенты прямой по двум точкам в пространстве.
        Принимает точку в виде массива 1x3 объекта класса numpy.array с тремя координатами [x, y, z]
        :param point1: [x1, y1, z1]
        :param point2: [x2, y2, z2]
        :return: None
        """
        if np.shape(point1)[0] == 2:
            self._c = 0
            p3 = 0
        else:
            self._c = point1[2]
            p3 = point2[2] - point1[2]
        self._a = point1[0]
        self._b = point1[1]

        # TODO: проверка на нуль
        p1 = point2[0] - point1[0]
        p2 = point2[1] - point1[1]

        if p1 == 0 and p2 == 0 and p3 == 0:
            logger.error("Создать линию из двух одинаковых точек нельзя")
        else:
            mod_N = sqrt(p1 ** 2 + p2 ** 2 + p3 ** 2)
            # Проверка на равенство длины вектора нормали единице
            if mod_N != 1.0:
                p1 = p1 / mod_N
                p2 = p2 / mod_N
                p3 = p3 / mod_N
            self.p1 = p1
            self.p2 = p2
            self.p3 = p3
        if np.shape(point1)[0] == 3:
            self.point1 = point1
            self.point2 = point2
            self.border_x = np.array([point1[0], point2[0]])
            self.border_y = np.array([point1[1], point2[1]])
            self.border_z = np.array([point1[2], point2[2]])

        else:
            self.point1 = np.array([point1[0], point1[1], 0])
            self.point2 = np.array([point2[0], point2[1], 0])
            self.border_x = np.array([point1[0], point2[0]])
            self.border_y = np.array([point1[1], point2[1]])
            self.border_z = np.array([0, 0])
        self.border_x.sort()
        self.border_y.sort()
        self.border_z.sort()

    def point_belongs_to_the_segment(self, point):
        if self.point_belongs_to_the_line(point):
            if self.inorno(point):
                return True
            else:
                return False
        else:
            return False

    def inorno(self, coordinate):
        """
        Функция проверяет, находится ли точка с прямой внутри заданного сегмента. Производится проверка на нулевой
         отрезок, если отрезок состоит из двух одинаковых точек, то смысла искать точку в нулевом отрезке нет.
        :param coordinate: [x, y, z]
        :return: True - принадлежит отрезку, False - не принадлежит отрезку
        """
        if (self.border_x[0] <= coordinate[0] <= self.border_x[1] and self.border_y[0] <= coordinate[1] <=
                self.border_y[1] and self.border_z[0] <= coordinate[2] <=
                self.border_z[1]):
            return True
        elif self.point1[0] == self.point2[0] and self.point1[1] == self.point2[1] and self.point1[2] == self.point2[2]:
            logger.debug("Нулевой отрезок")
        else:
            return False

    def get_points(self):
        return np.vstack([self.point1, self.point2])
    def show(self, ax):
        vT = self.get_points().T
        ax.plot(vT[0], vT[1], vT[2], 'm', color=self.color, linewidth=self.linewidth)
    # def lsftp(self, triangle, plane):
    #     '''
    #      Line segment from triangle and plane или сокращенно lsftp
    #     :return:
    #     '''
    #     pat = position_analyze_of_triangle(triangle.triangle_array())
    #     # if pat == 2:
    #     #     point1 = point_from_plane_line_intersection()
