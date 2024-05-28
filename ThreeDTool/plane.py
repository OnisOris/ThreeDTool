import matplotlib.pyplot as plt
from .twoDTool import *
from .line import Line_segment, Line
from .threeDTool import *


class Plane:
    # Коэффициенты уравнения плоскости a, b, c, d вида
    # ax + by + cz + d = 0
    def __init__(self, a=0, b=0, c=1, d=0):
        self.__a = a
        self.__b = b
        self.__c = c
        self.__d = d

    #
    def create_plane3(self, matrix):
        '''

        :param matrix:
        :return:
        '''
        G = np.array([[-1], [-1], [-1]])
        abc = np.dot(np.linalg.inv(matrix), G)
        coefficients = np.array([abc.T[0][0], abc.T[0][1], abc.T[0][2], 1])
        self.__a = coefficients[0]
        self.__b = coefficients[1]
        self.__c = coefficients[2]
        self.__d = coefficients[3]
        return coefficients
        # ^ y
        # |

    @property
    def a(self):
        return self.__a

    @property
    def b(self):
        return self.__b

    @property
    def c(self):
        return self.__c

    @property
    def d(self):
        return self.__d

    @a.setter
    def a(self, a):
        self.__a = a

    @b.setter
    def b(self, b):
        self.__b = b

    @c.setter
    def c(self, c):
        self.__c = c

    @d.setter
    def d(self, d):
        self.__d = d

    def get_N(self):
        """
        Возвращает координаты вектора нормали плоскости
        :return: np.array([a, b, c])
        """
        return np.array([self.__a, self.__b, self.__c])

    def create_plane_from_triangle(self, triangle, point=1, create_normal=False) -> None:
        """
        Данная функция принимает массив 4x3. Строка 1 - координаты вектора нормали (пишутся координаты только второй
        точки, первая исходит из нуля).
        Строки 2, 3, 4 - координаты вершин треугольника формата [x, y, z]
        На основе четырех точек создается плоскость и коэффициенты a, b, c, d записываются в поля объекта класса Plane
        Согласно параграфу 123 "Уравнение плоскости", плоскость, проходящая через точку M_0 (x_0, y_0, z_0) и
        перпендикулярная вектору N{a, b, c}, представляется уравнением:
        A(x - x_0) + B(y - y_0) + C(z - z_0) = 0 или Ax + By + Cz + D = 0,
        где D = -Ax_0 - By_o - Cz_0
        Поэтому мы берем первую вершину треугольника (по умолчанию point=1) и вектор нормали и на основе
        него создаем уравнение плоскости.
        Если create_normal = True, то это значит, что на вход идет матрица 3x3 с вершинами треугольника, тогда вектор
        нормали vector_N создается автоматически
        :return: None
        """
        if create_normal:
            vector_N = normal_of_triangle(triangle[0], triangle[1], triangle[2])
        else:
            vector_N = triangle[0]
        mod = sqrt(vector_N[0] ** 2 + vector_N[1] ** 2 + vector_N[2] ** 2)
        # Из-за неточного экспорта в STL и вычислений в Python модуль не будет точно равен 1,
        # но должен быть примерно равен 1
        if mod != 1:
            a, b, c = vector_N[0] / mod, vector_N[1] / mod, vector_N[2] / mod
        else:
            a, b, c = vector_N[0], vector_N[1], vector_N[2]
        first_point = triangle[point]
        self.__a, self.__b, self.__c = a, b, c
        #  Вычисление коэффициента D
        self.__d = - self.__a * first_point[0] - self.__b * first_point[1] - self.__c * first_point[2]

    ###################
    #       0         # hight
    ################### ->
    # lenth          x
    def show(self):
        # TODO: при параллельности плоскости оси z все ломается, нужно поменять способ отображения
        hight = 20  # Высота прямоугольника
        lenth = 20  # Длина прямоугольника
        x1 = -lenth / 2
        y1 = hight / 2
        z1 = self.projection_z(x1, y1)
        if z1 == "Uncertainty z":
            z1 = hight / 2
        point1 = np.array([x1, y1, z1])
        x2 = lenth / 2
        y2 = hight / 2
        z2 = self.projection_z(x2, y2)
        if z2 == "Uncertainty z":
            z2 = hight / 2
        point2 = np.array([x2, y2, z2])
        x3 = lenth / 2
        y3 = -hight / 2
        z3 = self.projection_z(x3, y3)
        if z3 == "Uncertainty z":
            z3 = -hight / 2
        point3 = np.array([x3, y3, z3])
        x4 = -lenth / 2
        y4 = -hight / 2
        z4 = self.projection_z(x4, y4)
        if z4 == "Uncertainty z":
            z4 = -hight / 2
        point4 = np.array([x4, y4, z4])

        matrix_x_y_z = self.full_vstack([point1, point2, point3, point4, point1]).T
        points = np.array([[x1, y1, 0], [x2, y2, 0], [x3, y3, 0], [x4, y4, 0], [x1, y1, 0]])
        matrix_points = self.full_vstack(points).T
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel("X", fontsize=15, color='red')
        ax.set_ylabel("Y", fontsize=15, color='green')
        ax.set_zlabel("Z", fontsize=15, color='blue')

        ax.plot(matrix_x_y_z[0], matrix_x_y_z[1], matrix_x_y_z[2], c='r')
        ax.plot(matrix_points[0], matrix_points[1], matrix_points[2], c='g')

        points_proj = np.array([[points[0], point1],
                                [points[1], point2],
                                [points[2], point3],
                                [points[3], point4]])
        figures = []
        for i in range(4):
            figures.append(ax.plot(points_proj[i].T[0], points_proj[i].T[1], points_proj[i].T[2], c='b'))
        plt.show()

    def projection_z(self, point_x, point_y):

        """
        Данная функция берет координаты плоскости x и y, и по ним ищет точку z на плоскости класса Plane.
        Рассмотрено три случая:
        1) Только c = 0, тогда точка z на плоскости может быть любой в координатах x, y.
        Поэтому ее можно задать вручную.
        2) b и a = 0, тогда плоскость параллельна осям y и x, а z = const, поэтому для нахождения z достаточно взять
        точку O(0, 0, z), тогда z = -D/C
        3) Нормальный вариант, когда ни один из коэффициентов не равен нулю.
        :param point_x:
        :param point_y:
        :return: Point z or "Uncertainty"
        """
        if self.__c == 0:
            return "Uncertainty z"
        elif self.__b == 0 and self.__a == 0:
            return -self.__d / self.__c
        else:
            point_z = (-self.__a * point_x - self.__b * point_y - self.__d) / self.__c
            return point_z

    def projection_y(self, point_x, point_z):

        """
        Данная функция берет координаты плоскости x и z, и по ним ищет точку y на плоскости класса Plane.
        Рассмотрено три случая:
        1) Только b = 0, тогда точка y на плоскости может быть любой в координатах x, z.
        Поэтому ее можно задать вручную.
        2) c и a = 0, тогда плоскость параллельна осям z и x, а y = const, поэтому для нахождения y достаточно взять
        точку O(0, y, 0), тогда y = -D/B
        3) Нормальный вариант, когда ни один из коэффициентов не равен нулю.
        :param point_x:
        :param point_z:
        :return: Point y or "Uncertainty"
        """
        if self.__b == 0:
            return "Uncertainty y"
        elif self.__a == 0 and self.__c == 0:
            return -self.__d / self.__b
        else:
            point_y = (-self.__a * point_x - self.__c * point_z - self.__d) / self.__b
        return point_y

    def projection_x(self, point_y, point_z):

        """
        Данная функция берет координаты плоскости x и z, и по ним ищет точку y на плоскости класса Plane.
        Рассмотрено три случая:
        1) Только a = 0, тогда точка y на плоскости может быть любой в координатах y, z.
        Поэтому ее можно задать вручную.
        2) a и c = 0, тогда плоскость параллельна осям x и z, а x = const, поэтому для нахождения x достаточно взять
        точку O(x, 0, 0), тогда x = -D/B
        3) Нормальный вариант, когда ни один из коэффициентов не равен нулю.
        :param point_y:
        :param point_z:
        :return: Point x or "Uncertainty"
        """
        if self.__a == 0:
            return "Uncertainty x"
        elif self.__b == 0 and self.__c == 0:
            return -self.__d / self.__a
        else:
            point_x = (-self.__b * point_y - self.__c * point_z - self.__d) / self.__a
            return point_x

    def full_vstack(self, vector):
        entry_point = vector[0]
        for element in vector:
            entry_point = np.vstack([entry_point, element])
        return entry_point


class Triangle(Plane):
    """
    Класс треугольника. При инициализации объекта происходит проверка на количество составляющих.
    Если np.shape(vertexes)[0] == 3, то считается, что в класс подаются вершины треугольника. Тогда нормаль считается
    по принципу правого винта, где напрвление задает порядок вершин.
    Если np.shape(vertexes)[0] == 4, то 4я координата - вектор нормали

    """

    def __init__(self, vertexes, auto_create_normal=False):
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
    def vertex1(self):
        return self.__vertex1

    @property
    def vertex2(self):
        return self.__vertex2

    @property
    def vertex3(self):
        return self.__vertex3

    @property
    def normal(self):
        return self.__normal

    @property
    def barycenter(self):
        return self.__barycenter

    @vertex1.setter
    def vertex1(self, vertex1):
        self.__vertex1 = vertex1

    @vertex2.setter
    def vertex2(self, vertex2):
        self.__vertex2 = vertex2

    @vertex3.setter
    def vertex3(self, vertex3):
        self.__vertex3 = vertex3

    @normal.setter
    def normal(self, normal):
        self.__normal = normal

    @barycenter.setter
    def barycenter(self, barycenter):
        self.__barycenter = barycenter

    def set_barycenter(self):
        arr = np.array([self.__vertex1,
                        self.__vertex2,
                        self.__vertex3])
        xyz_mean = arr.mean(axis=0)
        self.__barycenter = xyz_mean

    def line_segments_create(self):
        line1 = Line_segment()
        line2 = Line_segment()
        line3 = Line_segment()
        line1.segment_create_from_points(self.__vertex1, self.__vertex2)
        line2.segment_create_from_points(self.__vertex2, self.__vertex3)
        line3.segment_create_from_points(self.__vertex3, self.__vertex1)
        self.__line_segments = np.array([line1, line2, line3])

    def show(self, ax) -> None:
        vT = self.get_vertexes()
        vT = np.vstack([vT, self.get_vertexes()[0]]).T
        ax.plot(vT[0], vT[1], vT[2])


    def get_vertexes(self):
        return np.array([self.__vertex1, self.__vertex2, self.__vertex3])
    def get_mean_vertexes(self):
        return np.array([self.__vertex1, self.__vertex2, self.__vertex3]).mean(axis=0)

    def triangle_array(self):
        return np.array([self.__normal, self.__vertex1, self.__vertex2, self.__vertex3])

    def point_analyze(self, point: np.ndarray):
        """
        Функция принимает точку и проверяет, находится ли точка внутри границ треугольника в трехмерном пространстве
        путем подсчета числа
        пересечений с границами треугольника.
        :param point: np.ndarray
        :return: bool
        """

        # проверка принадлежности точки плоскости треугольника.
        # print(np.round(point_in_plane(self, point), 10))
        # print(point_in_plane(self, self.barycenter))
        p_in_plane = np.allclose(point_in_plane(self, point), 0, atol=1e-6)
        if p_in_plane:
            line = Line()
            test_point = self.barycenter  #self.__vertex1
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

