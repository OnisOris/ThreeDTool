from __future__ import annotations

from typing import Any

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
            self._vertices = vertices
        else:
            self._vertices = np.array(vertices)
        self._barycenter = np.array([])
        self._line_segments = np.array([])
        self.set_barycenter()
        self.line_segments_create()

    @property
    def barycenter(self) -> ndarray:
        return self._barycenter

    @property
    def vertices(self) -> ndarray:
        return self._vertices

    @property
    def line_segments(self) -> ndarray:
        return self._line_segments

    def get_closed_vertices(self) -> np.ndarray:
        """
        Функция соединяет последнюю и первую вершину
        :return: np.ndarray
        """
        return np.vstack([self._vertices, self._vertices[0]])

    def line_segments_create(self) -> None:
        """
        Создает отрезки из вершин фигуры
        :return: None
        """
        for i, item in enumerate(self._vertices):
            segment = Line_segment()
            if i == np.shape(self._vertices)[0] - 1:
                segment.segment_create_from_points(item, self._vertices[0])

            else:
                segment.segment_create_from_points(item, self._vertices[i + 1])
            self._line_segments = np.hstack([self._line_segments, segment])

    def get_line_segments(self) -> ndarray[Any, dtype[Any]]:
        """
        Возвращает отрезки
        :return: np.ndarray
        """
        return self._line_segments

    def set_barycenter(self) -> None:
        """
        Вычисляет барицентр фигуры
        :return: None
        """
        arr = self._vertices.T
        xyz_mean = arr.mean(axis=1)
        self._barycenter = xyz_mean

    def show(self, ax) -> None:
        """
        Функция отображения фигуры
        :return: None
        """
        for segment in self._line_segments:
            segment.color = 'green'
            segment.show(ax)

    def intersection_analyze(self, polygon: Polygon):
        """
        Функция находит точки пересечения с входящим многоугольником
        """
        from .threeDTool import point_from_segment_segment_intersection
        points = np.array([0, 0, 0])
        for segment in self._line_segments:
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

    def polygons_position_analyze(self, polygon: Polygon):
        """
        Функция определяет варианты расположений между собой многоугольников:
        a) В одной плоскости:
            0) Многоугольник совпадает
            1) Многоугольник пересекает второй
            2) Многоугольник внутри другого
            3 Многоугольники не пересекаются
        b) В разных плоскостях:
            4) Многоугольники параллельны и не пересекаются
            5) Многоугольники не параллельны и не пересекаются
            6) Многоугольники не параллельны и пересекаются частично
            7) Изучаемый многоугольник "входит" в self многоугольник
            8) Изучаемый многоугольник обромляет self многоугольник (ситуация противоположна 6)
            9) Многоугольники не параллельны и пересекаются "книгой"
        """
        from .threeDTool import position_analyzer_of_plane_plane
        from .plane import Plane
        # Если это одинаковые многоугольники, то возвращаем 3
        if self.polygons_equal(polygon):
            return 0
        else:
            plane1 = Plane()
            plane2 = Plane()
            plane1.create_plane_from_triangle(self._vertices[0:3], create_normal=True)
            plane2.create_plane_from_triangle(polygon.vertices[0:3], create_normal=True)
            var = position_analyzer_of_plane_plane(plane1, plane2)
            # 1 - если плоскости совпадают
            if var == 1:
                # Двухмерный случай

                # Смотрим пересечения
                var2 = self.intersection_analyze(polygon)
                if var2 is not None:
                    return 'a', 1
                else:
                    var3 = np.array([])
                    for v in polygon.vertices:
                        var3 = np.hstack([var3, self.point_analyze(v)])
                    if np.all(var3):
                        return 'a', 2
                    else:
                        return 'a', 3
            elif var == 0:
                # Трехмерный случай
                inters_point = np.array([])

            elif var == 2 or var == 3:
                return 'b', 4

    def polygons_equal(self, polygon: Polygon) -> bool:
        """
        В данной функции проверяется равенство многоугольников
        Для начала проверяется равенство размерности вершин, если размерность не одинаковая, то многоугольники не равны,
        если размерности равны, находятся две одинаковые вершины и массив второго прямоугольника пересортировывается,
        начиная с этой вершины. Если после сортировки массивы оказываются одинаковы, то многоугольники одинаковы.
        :param polygon: изучаемый полигон
        :type polygon: Polygon
        :return: bool | None
        """
        vert1 = self.vertices
        vert2 = polygon.vertices
        sort_vert = vert1[0]
        indices = np.where((np.isclose(sort_vert, vert2, 1e-6)).all(axis=1))[0]
        if self.vertices.shape != polygon.vertices.shape:
            return False
        else:
            if indices.shape == (0,):
                return False
            elif indices.shape == (1,):
                vert2 = np.roll(vert2, -indices[0], axis=0)
                vert2_rev = np.vstack([vert2[0], vert2[-1:0:-1]])
            else:
                raise Exception("Пришел сломанный многоугольник")
        if np.allclose(vert1, vert2, 1e-6) or np.allclose(vert1, vert2_rev, 1e-6):
            return True
        else:
            return False

    def point_analyze(self, point: np.ndarray) -> bool:
        """
        Функция принимает точку и проверяет, находится ли точка внутри границ многогранника путем подсчета числа
        пересечений с границами многогранника.
        :param point: Точка типа [x, y, z]
        :type point: np.ndarray
        :return: bool
        """
        from .threeDTool import point_comparison, point_from_beam_segment_intersection
        if point is None:
            return False
        line = Line()
        tets_point = np.array(self._barycenter)
        if point_comparison(point, self._barycenter):
            tets_point += 0.01

        line.line_create_from_points(point, tets_point)
        arr = np.array([[0, 0, 0]])
        for i, item in enumerate(self._line_segments):
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
        test_point = np.array(self._barycenter)
        line.line_create_from_points(point, test_point)
        arr = np.array([[0, 0, 0]])
        for i, item in enumerate(self._line_segments):
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

    def points_from_polygon_polygon_intersection(self, polygon: Polygon) -> ndarray[Any, dtype[Any]] | None:
        """
        Функция возвращает точки пересечения входящего полигона с self полигоном
        :param polygon: изучаемый полигон
        :type polygon: Polygon
        :return: ndarray[Any, dtype[Any]] | None
        """
        from .plane import Plane
        from .threeDTool import (point_from_plane_segment_intersection, position_analyzer_of_plane_plane,
                                 point_from_segment_segment_intersection)
        if self.polygons_equal(polygon):
            return self.vertices
        else:
            plane1 = Plane()
            plane1.create_plane_from_triangle(self._vertices[0:3], create_normal=True)
            plane2 = Plane()
            plane2.create_plane_from_triangle(polygon._vertices[0:3], create_normal=True)
            points = np.array([]).reshape(0, 3)
            var = position_analyzer_of_plane_plane(plane1, plane2)
            points_return = np.array([]).reshape(0, 3)
            if var == 0:
                for segment in polygon._line_segments:
                    point_in = point_from_plane_segment_intersection(segment, plane1)
                    if point_in is not None:
                        points = np.vstack([points, point_in])
                points = points
                if points.shape == (0, 3):
                    return None
                else:
                    for point in points:
                        var = self.point_analyze(point)
                        if var:
                            points_return = np.vstack([points_return, point])
                if points_return.shape != (0, 3):
                    return points_return
                else:
                    return None
            elif var == 1:
                for segment in polygon._line_segments:
                    for self_segment in self._line_segments:
                        point_int = point_from_segment_segment_intersection(segment, self_segment)
                        if point_int is not None:
                            points_return = np.vstack([points_return, point_int])
            if points_return.shape == (0, 3):
                return None
            else:
                return points_return


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
            tets_point += 0.001

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
