from __future__ import annotations
from typing import List, Any

import numpy as np
from numpy import ndarray, dtype, generic

from .line import Line_segment, Line


class Polygon:
    """
    Данный класс представляет собой многогранник, состоящий из координат вершин
    """

    def __init__(self, vertices: np.ndarray):
        """
        :param vertices: Вершины типа [[x1, y1, z1], ......., [xn, yn, zn]]
        :type vertices: np.ndarray
        """
        if isinstance(vertices, np.ndarray):
            self.__vertices = vertices
        else:
            self.__vertices = np.array(vertices)
        self.__barycenter = np.array([])
        self.__line_segments = np.array([])
        self.set_barycenter()
        self.line_segments_create()

    @property
    def barycenter(self) -> ndarray:
        return self.__barycenter

    def get_closed_vertices(self) -> np.ndarray:
        """
        Функция соединяет последнюю и первую вершину
        :return: np.ndarray
        """
        return np.vstack([self.__vertices, self.__vertices[0]])

    def line_segments_create(self) -> None:
        """
        Создает отрезки из вершин фигуры
        :return: None
        """
        for i, item in enumerate(self.__vertices):
            segment = Line_segment()
            if i == np.shape(self.__vertices)[0] - 1:
                segment.segment_create_from_points(item, self.__vertices[0])

            else:
                segment.segment_create_from_points(item, self.__vertices[i + 1])
            self.__line_segments = np.hstack([self.__line_segments, segment])

    def get_line_segments(self) -> ndarray[Any, dtype[Any]]:
        """
        Возвращает отрезки
        :return: np.ndarray
        """
        return self.__line_segments

    def set_barycenter(self) -> None:
        """
        Вычисляет барицентр фигуры
        :return: None
        """
        arr = self.__vertices.T
        xyz_mean = arr.mean(axis=1)
        self.__barycenter = xyz_mean

    def show(self, ax) -> None:
        """
        Функция отображения фигуры
        :return: None
        """
        for segment in self.__line_segments:
            segment.color = 'green'
            segment.show(ax)

    def intersection_analyze(self, polygon: Polygon):
        """
        Функция находит точки пересечения с входящим прямоугольником
        """
        from .threeDTool import point_from_segment_segment_intersection
        points = np.array([0, 0, 0])
        for segment in self.__line_segments:
            for segment_out in polygon.get_line_segments():
                point = point_from_segment_segment_intersection(segment_out, segment)
                if point is not None:
                    points = np.vstack([points, point])
        if np.shape(points) != (3,):
            points = points[1:]
        else:
            return None
        if points.shape[0] > 0:
            return points
        else:
            return None
    #
    # def polygons_position_analyze(self, polygon: Polygon):
    #     """
    #     Функция определяет варианты расположений между собой многоугольников:
    #     a) В одной плоскости:
    #         1) Многоугольник внутри другого
    #         2) Многоугольник пересекает второй
    #         3) Многоугольник совпадает
    #     b) В разных плоскостях:
    #         1) Многоугольники параллельны
    #         2) Многоугольники не параллельны и не пересекаются
    #         3) Многоугольники не параллельны и пересекаются частично
    #         4) Изучаемый многоугольник "входит" в self многоугольник
    #         5) Изучаемый многоугольник обромляет self многоугольник (ситуация противоположна 4)
    #         6) Многоугольники не параллельны и пересекаются "книгой"
    #     """
    # from .threeDTool import Plane
    # pass

    def point_analyze(self, point: np.ndarray) -> bool:
        """
        Функция принимает точку и проверяет, находится ли точка внутри границ многогранника путем подсчета числа
        пересечений с границами многогранника.
        :param point: Точка типа [x, y, z]
        :type point: np.ndarray
        :return: bool
        """
        from .threeDTool import point_comparison, point_from_beam_segment_intersection
        line = Line()
        tets_point = np.array(self.__barycenter)
        if point_comparison(point, self.barycenter):
            tets_point += 1

        line.line_create_from_points(point, tets_point)
        arr = np.array([[0, 0, 0]])
        for i, item in enumerate(self.__line_segments):
            p = np.array(point_from_beam_segment_intersection(line, item))
            if np.shape(p) == (3,):
                arr = np.vstack([arr, p])
        arr = arr[1:np.shape(arr)[0]]
        if np.shape(point)[0] == 2:
            point = np.hstack([point, 0])
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

    def point_of_intersection(self, point: np.ndarray) -> ndarray:
        """
        Функция принимает точку и возвращает точки пересечения луча с фигурой.
        :param point: Изучаемая точка
        :type point: np.ndarray
        :return: bool
        """
        from .threeDTool import point_from_beam_segment_intersection, point_comparison
        line = Line()
        tets_point = np.array(self.__barycenter)
        line.line_create_from_points(point, tets_point)
        arr = np.array([[0, 0, 0]])
        for i, item in enumerate(self.__line_segments):
            p = np.array(point_from_beam_segment_intersection(line, item))
            if np.shape(p) == (3,):
                arr = np.vstack([arr, p])
        arr = arr[1:np.shape(arr)[0]]
        if np.shape(point)[0] == 2:
            point = np.hstack([point, 0])
        arr = np.unique(arr, axis=0)
        idx = np.array([])
        for i, item in enumerate(arr):
            if point_comparison(item, point):
                idx = np.hstack([idx, i])
        if np.shape(idx)[0] != 0:
            idx = idx.astype("int")
            arr = np.delete(arr, idx, axis=0)
        return arr


class Polygon_2D:
    """
    Класс многоугольника в 2D пространстве
    """

    def __init__(self, vertices: np.ndarray):
        """
        :param vertices: Вершины типа [[x1, y1], ......., [xn, yn]]
        :type vertices: np.ndarray
        """
        self.__vertices = vertices
        self.__barycenter = np.array([])
        self.__line_segments = []
        self.set_barycenter()
        self.line_segments_create()

    @property
    def barycenter(self) -> ndarray:
        return self.__barycenter

    def get_closed_vertices(self):
        return np.vstack([self.__vertices, self.__vertices[0]])

    def line_segments_create(self):
        for i, item in enumerate(self.__vertices):
            segment = Line_segment()
            if i == np.shape(self.__vertices)[0] - 1:
                segment.segment_create_from_points(item, self.__vertices[0])

            else:
                segment.segment_create_from_points(item, self.__vertices[i + 1])
            self.__line_segments = np.hstack([self.__line_segments, segment])

    def get_line_segments(self):
        return self.__line_segments

    def set_barycenter(self):
        arr = self.__vertices.T
        xyz_mean = arr.mean(axis=1)
        self.__barycenter = xyz_mean

    def point_analyze(self, point: np.ndarray) -> bool:
        """
        Функция принимает точку и проверяет, находится ли точка внутри границ многогранника путем подсчета числа
        пересечений с границами многогранника.
        :param point: Точка типа [x, y, z]
        :type point: np.ndarray
        :return: bool
        """
        from .threeDTool import point_comparison, point_from_beam_segment_intersection
        line = Line()
        tets_point = np.array(self.__barycenter)
        if point_comparison(point, self.barycenter):
            tets_point += 1

        line.line_create_from_points(point, tets_point)
        arr = np.array([[0, 0, 0]])
        for i, item in enumerate(self.__line_segments):
            p = np.array(point_from_beam_segment_intersection(line, item))
            if np.shape(p) == (3,):
                arr = np.vstack([arr, p])
        arr = arr[1:np.shape(arr)[0]]
        if np.shape(point)[0] == 2:
            point = np.hstack([point, 0])
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

    def point_of_intersection(self, point: np.ndarray) -> ndarray[Any, dtype[generic | generic | Any]]:
        """
       Функция принимает точку и возвращает точки пересечения луча с фигурой
       :param point: Точка типа [x, y, z]
       :type point: np.ndarray
       :return: bool
       """
        from .threeDTool import point_from_beam_segment_intersection, point_comparison
        line = Line()
        tets_point = np.array(self.__barycenter)
        line.line_create_from_points(point, tets_point)
        arr = np.array([[0, 0, 0]])
        for i, item in enumerate(self.__line_segments):
            p = np.array(point_from_beam_segment_intersection(line, item))
            if np.shape(p) == (3,):
                arr = np.vstack([arr, p])
        arr = arr[1:np.shape(arr)[0]]
        if np.shape(point)[0] == 2:
            point = np.hstack([point, 0])
        arr = np.unique(arr, axis=0)
        idx = np.array([])
        for i, item in enumerate(arr):
            if point_comparison(item, point):
                idx = np.hstack([idx, i])
        if np.shape(idx)[0] != 0:
            idx = idx.astype("int")
            arr = np.delete(arr, idx, axis=0)
        return arr
