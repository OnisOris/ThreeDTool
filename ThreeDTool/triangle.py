# from __future__ import annotations
from .plane import Plane
# from .line import Line_segment, Line
import numpy as np


class Triangle(Plane):
    """
    Класс треугольника. При инициализации объекта происходит проверка на количество составляющих.
    Если np.shape(vertexes)[0] == 3, то считается, что в класс подаются вершины треугольника. Тогда нормаль считается
    по принципу правого винта, где направление задает порядок вершин.
    Если np.shape(vertexes)[0] == 4, то 4я координата - вектор нормали
    """

    def __init__(self, vertexes: np.ndarray, auto_create_normal: bool = False):
        """
        :param vertexes: Массив вершин треугольников 4x3 или 3x3
        :type vertexes: np.ndarray
        """
        from .threeDTool import normal_of_triangle
        super().__init__()
        self.__line_segments = None
        if np.shape(vertexes)[0] == 3 or auto_create_normal:
            self.__vertex1 = np.array(vertexes[0])
            self.__vertex2 = np.array(vertexes[1])
            self.__vertex3 = np.array(vertexes[2])
            self.__normal = normal_of_triangle(self.__vertex1, self.__vertex2, self.__vertex3)
            self.create_plane_from_triangle(np.array([self.__normal, self.__vertex1, self.__vertex2, self.__vertex3]))
        if np.shape(vertexes)[0] == 4:
            self.__vertex1 = np.array(vertexes[1])
            self.__vertex2 = np.array(vertexes[2])
            self.__vertex3 = np.array(vertexes[3])
            mod = np.linalg.norm(vertexes[0])
            self.__normal = np.array(vertexes[0] / mod)
            self.create_plane_from_triangle(np.array([self.__normal, self.__vertex1, self.__vertex2, self.__vertex3]))
        self.__barycenter = None
        self.set_barycenter()
        self.line_segments_create()

    @property
    def vertex1(self) -> np.ndarray:
        return self.__vertex1

    @property
    def vertex2(self) -> np.ndarray:
        return self.__vertex2

    @property
    def vertex3(self) -> np.ndarray:
        return self.__vertex3

    @property
    def normal(self) -> np.ndarray:
        return self.__normal

    @property
    def barycenter(self) -> np.ndarray:
        return self.__barycenter

    @vertex1.setter
    def vertex1(self, vertex1) -> None:
        self.__vertex1 = vertex1

    @vertex2.setter
    def vertex2(self, vertex2) -> None:
        self.__vertex2 = vertex2

    @vertex3.setter
    def vertex3(self, vertex3) -> None:
        self.__vertex3 = vertex3

    @normal.setter
    def normal(self, normal) -> None:
        self.__normal = normal

    @barycenter.setter
    def barycenter(self, barycenter) -> None:
        self.__barycenter = barycenter

    def set_barycenter(self) -> None:
        """
        Устанавливает барицентр для треугольника из вершин треугольника
        :return: None
        """
        arr = np.array([self.__vertex1,
                        self.__vertex2,
                        self.__vertex3])
        xyz_mean = arr.mean(axis=0)
        self.__barycenter = xyz_mean

    def line_segments_create(self) -> None:
        """
        Создает отрезки на основе вершин треугольника
        :return: None
        """
        line1 = Line_segment()
        line2 = Line_segment()
        line3 = Line_segment()
        line1.segment_create_from_points(self.__vertex1, self.__vertex2)
        line2.segment_create_from_points(self.__vertex2, self.__vertex3)
        line3.segment_create_from_points(self.__vertex3, self.__vertex1)
        self.__line_segments = np.array([line1, line2, line3])

    def show(self, ax) -> None:
        """
        Функция для отображения треугольника
        :param ax: Объект сцены matplotlib
        :type ax: matplotlib.axes.Axes
        :return: None
        """
        vT = self.get_vertexes()
        vT = np.vstack([vT, self.get_vertexes()[0]]).T
        ax.plot(vT[0], vT[1], vT[2])

    def get_vertexes(self) -> np.ndarray:
        """
        Функция возвращает вершины треугольника размерностью 3x3
        :return: np.ndarray
        """
        return np.array([self.__vertex1, self.__vertex2, self.__vertex3])

    def get_mean_vertexes(self) -> np.ndarray:
        """
        Функция возвращает среднюю координату типа [x_mean, y_mean, z_mean]
        :return: np.ndarray
        """
        return np.array([self.__vertex1, self.__vertex2, self.__vertex3]).mean(axis=0)

    def triangle_array(self) -> np.ndarray:
        """
        Возвращает массив треугольника с нормалью 4x3
        :return: np.ndarray
        """
        return np.array([self.__normal, self.__vertex1, self.__vertex2, self.__vertex3])

    def point_analyze(self, point: np.ndarray):
        """
        Функция принимает точку и проверяет, находится ли точка внутри границ треугольника в трехмерном пространстве
        путем подсчета числа
        пересечений с границами треугольника.
        :param point: np.ndarray
        :return: bool
        """
        from .threeDTool import point_in_plane, point_comparison, point_from_beam_segment_intersection
        # проверка принадлежности точки плоскости треугольника.
        p_in_plane = np.allclose(point_in_plane(self, point), 0, atol=1e-6)
        if p_in_plane:
            line = Line()
            test_point = self.barycenter  # self.__vertex1
            if point_comparison(point, self.barycenter):
                test_point = self.vertex1
            # создание линии из оцениваемой точки в барицентр или в одну из вершин
            line.line_create_from_points(point, test_point)
            arr = np.array([[0, 0, 0]])
            for i, item in enumerate(self.__line_segments):
                p = np.array(point_from_beam_segment_intersection(line, item))
                if np.shape(p) == (3,):
                    arr = np.vstack([arr, p])
            arr = arr[1:np.shape(arr)[0]]
            arr = np.unique(arr, axis=0)
            idx = np.array([])
            for i, item in enumerate(arr):
                if point_comparison(item, point):
                    idx = np.hstack([idx, i])
            if np.shape(idx)[0] != 0:
                idx = idx.astype("int")
                arr = np.delete(arr, idx, axis=0)
            var = (np.shape(arr)[0]) % 2
            if var == 0:
                return False
            else:
                return True
        else:
            return False
