import numpy as np
from .line import Line_segment, Line
class Polygon:
    def __init__(self, vertices: np.ndarray) -> None:
        self.__vertices = vertices
        self.__barycenter = np.array([])
        self.__line_segments = []
        self.set_barycenter()
        self.line_segments_create()

    @property
    def barycenter(self):
        return self.__barycenter

    def get_closed_vartices(self):
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
    def show(self, ax) -> None:
        for segment in self.__line_segments:
            segment.color = 'green'
            segment.show(ax)


    def point_analyze(self, point: np.ndarray):
        """
        Функция принимает точку и проверяет, находится ли точка внутри границ многогранника путем подсчета числа
        пересечений с границами многогранника.
        :param point: np.ndarray
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
    def point_of_intersection(self, point: np.ndarray):
        """
               Функция принимает точку и возвращает точки пересечения луча с фигурой
               :param point: np.ndarray
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
    def __init__(self, vertices: np.ndarray) -> None:
        self.__vertices = vertices
        self.__barycenter = np.array([])
        self.__line_segments = []
        self.set_barycenter()
        self.line_segments_create()

    @property
    def barycenter(self):
        return self.__barycenter

    def get_closed_vartices(self):
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

    def point_analyze(self, point: np.ndarray):
        """
        Функция принимает точку и проверяет, находится ли точка внутри границ многогранника путем подсчета числа
        пересечений с границами многогранника.
        :param point: np.ndarray
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
    def point_of_intersection(self, point: np.ndarray):
        """
               Функция принимает точку и возвращает точки пересечения луча с фигурой
               :param point: np.ndarray
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