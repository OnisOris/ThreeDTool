from __future__ import annotations
from typing import Tuple, Any
from numpy import ndarray, dtype
import matplotlib
from .line import Line, Line_segment
import numpy as np
from loguru import logger
from typing import Optional

log = False


def check_position_lines(line1: Line, line2: Line) -> int:
    """
    :param line1: Первая линия
    :type line1: Line
    :param line2: Вторая линия
    :type line2: Line
    :return: 0 - если линии не компланарны, 1 - если прямые компланарны параллельны, 2 - если прямые компланарны и
    не параллельны 3, если линии совпадают
    """
    cross = np.round(np.linalg.norm(np.cross(line1.coeffs()[3:6], line2.coeffs()[3:6])), 6)
    if ((np.array_equal(line1.coeffs()[0:3], line2.coeffs()[0:3]) and
         np.linalg.matrix_rank(np.array([line1.coeffs()[3:6], line2.coeffs()[3:6], line2.coeffs()[3:6]])) == 2)
            and cross == 0):
        return 3
    else:
        line3 = Line()
        line3.line_create_from_points(line1.coeffs()[0:3], line2.coeffs()[0:3])
        # Проверка на компланарность. Если определитель трех векторов равен нулю, то они находятся в одной плоскости.
        arr = np.array([line3.coeffs()[3:6],
                        line1.coeffs()[3:6],
                        line2.coeffs()[3:6]])
        var = np.round(np.linalg.det(arr), 6)

        if var == 0:
            if cross == 0:
                # прямые параллельны
                return 1
            else:
                # прямые не параллельны
                return 2
        else:
            # прямые не компланарны
            return 0


def point_from_line_line_intersection(line1, line2):
    """
    Функция возвращает точку пересечения двух линий. В случае, если линии не параллельны или не компланарны, то вернет
    False
    :param line1: Первая линия
    :type line1: Line
    :param line2: Вторая линия
    :type line2: Line
    :return: np.ndarray([x, y, z])
    """
    # Проверка на принадлежность одной плоскости
    var = check_position_lines(line1, line2)
    if var == 2:
        if np.allclose(line1.coeffs()[3:6], 0, atol=1e-8):
            return None
        elif np.allclose(line2.coeffs()[3:6], 0, atol=1e-8):
            return None
        if not np.allclose(line2.p1 * line1.p3, line2.p3 * line1.p1, atol=1e-8):
            # t_2^x
            t = ((line1.a * line1.p3 - line2.a * line1.p3 + line2.c * line1.p1 - line1.c * line1.p1) /
                 (line2.p1 * line1.p3 - line2.p3 * line1.p1))
        elif not np.allclose(line2.p2 * line1.p3, line2.p3 * line1.p2, atol=1e-8):
            # t_2^y
            t = ((line1.b * line1.p3 - line2.b * line1.p3 + line1.p2 * line2.c - line1.p2 * line1.c) /
                 (line2.p2 * line1.p3 - line2.p3 * line1.p2))
        else:
            # t_2^z
            t = ((line1.a * line1.p2 - line2.a * line1.p2 + line2.b * line1.p1 - line1.b * line1.p1) /
                 (line2.p1 * line1.p2 - line2.p2 * line1.p1))
        x = t * line2.p1 + line2.a
        y = t * line2.p2 + line2.b
        z = t * line2.p3 + line2.c
        return np.round(np.array([x, y, z]), 18)
    else:
        if log:
            logger.error("Прямые не пересекаются, либо совпадают")
        return False


def point_from_plane_line_intersection(line, plane) -> Optional[np.ndarray]:
    """
    Функция находит координаты точки пересечения линии line и плоскости plane.
    :param line: Объект линии
    :type line: Line
    :param plane: объект класса плоскости
    :type plane: Plane
    :return: np.ndarray[x, y, z] or None
    """
    # Проверка на параллельность линии плоскости
    vector_n = np.array([plane.a, plane.b, plane.c])
    vector_line = np.array([line.p1, line.p2, line.p3])
    if np.dot(vector_n, vector_line) != 0:
        line_abc = line.coeffs()[0:3]
        t = - np.sum([line_abc.dot(vector_n), plane.d]) / vector_n.dot(vector_line)
        arr = np.sum([vector_line.dot(t), line_abc], axis=0)
        return arr
    else:
        if log:
            logger.debug("Прямая параллельная плоскости")
        return None


def point_from_line_segment_intersection(line, segment):
    """
    Функция находит точку пересечения линии и отрезка. В случае, если линия и отрезок не пересекаются,
    то возвращается None
    :param line: Объект линии
    :type line: Line
    :param segment: Объект отрезка
    :type segment: Line_segment
    :return: np.ndarray[x, y, z] or None

    """
    point = point_from_line_line_intersection(line, segment)
    if point.__class__ != None.__class__:
        if segment.point_belongs_to_the_segment(point):
            return point
        else:
            return None
    else:
        return None


def point_from_beam_segment_intersection(beam, segment):
    """
    Функция возвращает точку пересечения луча и отрезка. Если луч ничего не пересекает, возвращается False.
    В качестве луча используется объект линии, где началом считается точка [a, b, c], а вектор направления [p1, p2, p3]
    :param beam: Луч, описываемый уравнением линии
    :type beam: Line
    :param segment: Объект отрезка
    :type segment: Line_segment
    :return: np.ndarray[x, y, z] or None
    """
    point = point_from_line_line_intersection(beam, segment)

    if point.__class__ == False.__class__:
        return False
    D = - beam.coeffs()[0:3].dot(beam.coeffs()[3:6])
    var = np.sum([beam.coeffs()[3:6].dot(point), D])
    if var >= 0 or np.allclose(var, 0, atol=1e-8):
        if point.__class__ != None.__class__:
            if segment.point_belongs_to_the_segment(point):
                return point
            else:
                return False
        else:
            return False
    else:
        return False


def point_from_segment_segment_intersection(segment1, segment2):
    """
    Функция возвращает точку пересечения двух отрезков
    :param segment1: Объект первого отрезка
    :type segment1: Line_segment
    :param segment2: Объект второго отрезка
    :type segment2: Line_segment
    :return: np.ndarray[x, y, z] or None
    """
    point = point_from_line_line_intersection(segment1, segment2)
    if isinstance(point, ndarray):
        if segment1.point_belongs_to_the_segment(point) and segment2.point_belongs_to_the_segment(point):
            return point
        else:
            return None
    else:
        return None


def max_min_points(triangles):
    """
    Функция принимает массив из координат треугольников и возвращает минимальные максимальные точки x, y, z в виде
    списка max = [x_max, y_max, z_max], min = [x_min, y_min, z_min]
    :param triangles: Массив 4x3, где первая строка - нормаль треугольника
    :type triangles: np.ndarray[float, int]
    :return: np.ndarray[x_max, y_max, z_max], np.ndarray[x_min, y_min, z_min]
    """
    x = np.array([])
    y = np.array([])
    z = np.array([])
    for i in range(triangles.shape[0]):
        x = np.append(x, triangles[:][i].T[0][1:4])
        y = np.append(y, triangles[:][i].T[1][1:4])
        z = np.append(z, triangles[:][i].T[2][1:4])
    max_xyz = np.array([np.max(x), np.max(y), np.max(z)])
    min_xyz = np.array([np.min(x), np.min(y), np.min(z)])
    return max_xyz, min_xyz


def position_analyzer_of_point_and_plane(point: list or np.ndarray, plane: Plane) -> int:
    """
    Функция принимает точку в виде списка [x, y, z] и плоскость класса Plane. Функция говорит, по какую сторону от
    плоскости лежит точка.
    1 - точка находится перед плоскостью (куда смотрит нормаль плоскости);
    0 - точка лежит на плоскости;
    -1 - точка лежит за плоскостью (в противоположную сторону от направления нормали).
    :param point:
    :type point: np.ndarray or list
    :param plane: Plane
    :return: 1, 0, -1
    """
    var = np.round(point_in_plane(plane, point), 8)
    if var > 0:
        return 1
    if var < 0:
        return -1
    else:
        return 0


def position_analyzer_of_line_and_plane(line: Line, plane: Plane):
    '''
    Функция анализирует положение линии относительно плоскости. Линия может быть: параллельна плоскости, лежать в ней,
    пересекать плоскость в точке.
    :param line: Объект линии
    :type line: Line
    :param plane: Объект плоскости
    :type plane: Plane
    :return: 0, если линия пренадлежит плоскости, 1, если линия параллельна плоскости и не принадлежит ей, 2, если
    линия не параллельна плоскости и пересекает ее в какой-то точке
    '''
    # Если var1 == 0 и var2 == 1, то линия либо в плоскости, если var1 != 0 и var2 == 1, то линия не в плоскости и
    # параллельна ей, если var2 != 1, то линия пересекает плоскость
    var1 = plane.a * line.a + plane.b * line.b + plane.c * line.c + plane.d
    var2 = np.linalg.norm(np.cross(line.coeffs()[3:6], plane.get_N()))
    if var1 == 0 and var2 == 1:
        return 0
    elif var1 != 0 and var2 == 1:
        return 1
    elif var2 != 1:
        return 2
    else:
        if log:
            logger.error("Что-то пошло не так, таких ситуаций в реальности не существует")


def position_analyze_of_triangle(triangle: list | np.ndarray, plane: Plane) -> (tuple[int, None] | tuple[int, Any] |
                                                                                tuple[int, ndarray[Any, dtype[Any]]]):
    """
    Функция принимает массив треугольников 4x3, где строка 1 - вектор нормали, строки 2-4 - это координаты вершин
    треугольников и плоскость класса Plane и говорит о местоположении треугольника относительно этой плоскости.
    Возвращает код в виде четырех чисел:
    2, None - треугольник пересекает плоскость (две вершины лежат в плоскости - считается за пересечение);
    1, None - треугольник находится перед плоскостью (куда смотрит нормаль плоскости);
    0, None - треугольник лежит в плоскости;
    -1, None - треугольник лежит за плоскости
    -2, [x, y, z] - только одна вершина треугольника лежит в плоскости
    (в противоположную сторону от направления нормали). Также возвращается точка вершины, которая лежит в плоскости.

    :param triangle: Массив треугольника
    :type triangle: np.ndarray
    :param plane: Объект плоскости
    :type plane: Plane
    :return: 2, 1, 0, -1
    """
    point1 = triangle[1]
    point2 = triangle[2]
    point3 = triangle[3]
    var1 = position_analyzer_of_point_and_plane(point1, plane)
    var2 = position_analyzer_of_point_and_plane(point2, plane)
    var3 = position_analyzer_of_point_and_plane(point3, plane)
    p1 = point_in_plane(plane, point1)
    p2 = point_in_plane(plane, point2)
    p3 = point_in_plane(plane, point3)
    if var1 == 1 and var2 == 1 and var3 == 1:
        return 1, None
    elif var1 == -1 and var2 == -1 and var3 == -1:
        return -1, None
    elif var1 == 0 and var2 == 0 and var3 == 0:
        return 0, None
    elif var1 == 0 and var2 == 1 and var3 == 1 or var1 == 0 and var2 == -1 and var3 == -1 \
            or var1 == 1 and var2 == 0 and var3 == 1 or var1 == -1 and var2 == 0 and var3 == -1 \
            or var1 == 1 and var2 == 1 and var3 == 0 or var1 == -1 and var2 == -1 and var3 == 0:
        if p1 == 0:
            return -2, point1
        elif np.allclose(p2, 0, atol=1e-5):
            return -2, point2
        elif p3 == 0:
            return -2, point3
        else:
            return -2, None  # Такого варианта не должно быть, если произошло - ошибка в вычислениях
    else:
        if (var1 == -1 and var2 == -1) or (var1 == 1 and var2 == 1) or (var1 == 0 and var2 == 0):
            return 2, np.array([[point1, point3],
                                [point2, point3]])
        elif (var2 == -1 and var3 == -1) or (var2 == 1 and var3 == 1) or (var2 == 0 and var3 == 0):
            return 2, np.array([[point2, point1],
                                [point3, point1]])
        elif (var1 == -1 and var3 == -1) or (var1 == 1 and var3 == 1) or (var1 == 0 and var3 == 0):
            return 2, np.array([[point1, point2], [point3, point2]])
        else:
            if log:
                logger.error("Такого случая не должно быть")


def point_in_plane(plane: Plane, point: list | np.ndarray) -> float:
    """
    Функция проверяет факт принадлежности точки плоскости
    :param plane: Объект плоскости
    :type plane: Plane
    :param point: Массив точки [x, y, z]
    :type point: list | np.ndarray
    :return: float | int
    """
    var = plane.a * point[0] + plane.b * point[1] + plane.c * point[2] + plane.d
    return var


def distance_between_two_points(point1: float | int, point2: float | int) -> float:
    """
    Данная функция берет две точки с прямой (одномерное пространство) и возвращает расстояние между ними.
    Примечание: принимает два числа float.
    :param point1: Первая точка типа x
    :type point1: int | float
    :param point2: Вторая точка типа x
    :type point2: float | int
    :return: float
    """
    array = np.sort(np.array([point1, point2]))
    if array[0] < 0 <= array[1]:
        return float(abs(array[0]) + array[1])
    elif array[0] >= 0:
        return float(array[1] - array[0])
    else:
        return float(abs(array[0]) - abs(array[1]))


def vector_from_two_points(point1, point2) -> np.ndarray:
    """
    Данная функция создает вектор из двух точек
    :param point1: Первая точка типа [x, y, z] или [x, y]
    :type point1: list or np.ndarray
    :param point2: Первая точка типа [x, y, z] или [x, y]
    :type point2: list or np.ndarray
    :return: np.ndarray
    """
    if np.shape(point1)[0] == 3:
        vector = np.array([point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2]])
    else:
        vector = np.array([point2[0] - point1[0], point2[1] - point1[1]])
    return vector


def normal_of_triangle(vertex1, vertex2, vertex3) -> np.ndarray:
    """
    Функция генерирует вектор нормали треугольника по правиле правого винта
    :param vertex1: Первая вершина треугольника
    :type vertex1: list or np.ndarray
    :param vertex2: Вторая вершина треугольника
    :type vertex2: list or np.ndarray
    :param vertex3: Третья вершина треугольника
    :type vertex3: list or np.ndarray
    :return: np.ndarray
    """
    vertex1 = np.array(vertex1)
    vertex2 = np.array(vertex2)
    vertex3 = np.array(vertex3)
    vector1 = vector_from_two_points(vertex1, vertex2)
    vector2 = vector_from_two_points(vertex1, vertex3)
    normal = np.cross(vector1, vector2)
    mod_normal = np.linalg.norm(normal)
    # Проверка на равенство длины вектора нормали единице
    if mod_normal != 1.0:
        normal = np.array([normal[0] / mod_normal, normal[1] / mod_normal, normal[2] / mod_normal])
    return normal


def null_vector(vector) -> bool:
    """
    Функция проверяет является ли вектор нулевым
    :param vector: Вектор типа [x, y, z] или [x, y]
    :type vector: list or np.ndarray
    :return: bool
    """
    if np.linalg.norm(vector) == 0:
        return True
    else:
        return False


def point_comparison(point1, point2):
    """
    Функция проверяет равенство двух точек. В случае равенства возвращает True, иначе False
    :param point1: Первая точка типа [x, y, z]
    :type point1: list or np.ndarray
    :param point2: Вторая точка типа [x, y, z]
    :type point2: list or np.ndarray
    :return: bool
    """
    n = 7
    point1 = np.round(point1, n)
    point2 = np.round(point2, n)

    if np.shape(point1)[0] == 2:
        point1 = np.hstack([point1, 0])
    if np.shape(point2)[0] == 2:
        point2 = np.hstack([point2, 0])
    if np.allclose(point1, point2, atol=1e-6):
        return True
    else:
        return False


def line_triangle_intersection(line: Line, triangle: Triangle) -> bool | np.ndarray:
    """
    Функция возвращает точку пересечения линии и треугольника
    :param line: Исследуемая линия
    :type line: Line
    :param triangle: Исследуемый треугольник
    :type triangle: Triangle
    :return: bool or np.ndarray
    """
    point = point_from_plane_line_intersection(line, triangle)
    if point is not None:
        if triangle.point_analyze(point):
            return point
        else:
            return False
    else:
        return False


def beam_triangle_intersection(beam: Line, triangle: Triangle) -> np.ndarray | None:
    """
    Данная функция находит пересечение луча и треугольника. Луч представлен объектом класса Line, так как в
    каноническом уравнении линии есть все для его описания (точка прохождение и направление)
    :param beam: Объект класса линии, представляющий собой луч в данной интерпретации
    :type beam: Line
    :param triangle: Объект класса треугольника
    :type triangle: Triangle
    """
    point = line_triangle_intersection(beam, triangle)
    if point is not None and point is not False:
        beam_abc = beam.coeffs()[0:3]
        beam_p = beam.coeffs()[3:6]
        D = - beam_abc.dot(beam_p)
        var = np.sum([beam_p.dot(point), D])
        if var >= 0:
            return point
        else:
            return None
    else:
        return None


def loxodrome(angle: float = 0,
              R: float = 70,
              count_of_rot: float = 17,
              step: float = 0.0025,
              point_n: ndarray = np.array([0, 0, 0])):
    """
    Функция генерирует локсодрому
    :param angle: Вертикальный угол
    :type angle: float
    :param R: Радиус локсодромы
    :type R: float
    :param count_of_rot: Число витков
    :type count_of_rot: int
    :param step: Количество шагов точность
    :type step: float
    :param point_n: Центральная точка локсодромы формата [x, y, z]
    :type point_n: np.ndarray or list
    :return: np.ndarray
    """
    v_angle_unit = np.pi / R / 2
    h_angle_unit = np.pi / R * count_of_rot * step
    xr = angle / 180 * np.pi
    xrc = np.cos(xr)
    xrs = np.sin(xr)
    total_rot = 0
    arr = np.array([0, 0, 0])
    i = -R
    while i <= R:
        x = np.cos(i * v_angle_unit) * R
        y = np.sin(i * v_angle_unit) * R
        v = [x * np.cos(total_rot), y, x * np.sin(total_rot)]
        pnt_y = v[1] * xrc - v[2] * xrs
        pnt_z = v[2] * xrc + v[1] * xrs
        arr = np.vstack([arr, [v[0], pnt_y, pnt_z]])
        total_rot += h_angle_unit
        i += step
    if arr.shape != (3,):
        arr = arr[1:np.shape(arr)[0]]
    return arr + point_n


def generate_loxodromes(r: float = 10.0,
                        r_c: float = 0,
                        layer_height: float = 0.2,
                        point_n: ndarray = np.array([0, 0, 0]),
                        steps: float = 0.0025) -> ndarray[Any, dtype[Any]]:
    """
    Функция создает вложенные друг в друга локсодромы
    :param r: Внешний радиус
    :type r: float
    :param r_c: Внутренний радиус
    :type r_c: float
    :param layer_height: Высота слоев или расстояние между локсодромами
    :type layer_height: float
    :param point_n: Центральная точка локсодром
    :type point_n: float
    :param steps: Точность
    :type steps: float
    :return: np.ndarray[Curve]
    """
    from .curve import Curve
    count_of_layer = r / layer_height - r_c / layer_height
    step = (r - r_c) / count_of_layer
    curves = np.array([])
    for i in range(int(count_of_layer)):
        curves = np.hstack([curves, Curve(loxodrome(R=r - i * step,
                                                    angle=90,
                                                    count_of_rot=17,
                                                    step=steps,
                                                    point_n=point_n))])
    return curves


def matrix_dot_all(self, array_matrix):
    T0_2 = array_matrix[0].dot(array_matrix[1])
    return T0_2


def matrix_create(cja: list or ndarray, DH: dict):
    """
    Функция создает матрицы преобразования координат на основе DH параметров
    :param cja: Углы поворота типа [angle_1, ....., angle_n]
    :type cja: list or ndarray
    :param DH: Словарь с DH параметрами
    :type DH: dict
    :return: ndarray
    """
    from numpy import (sin, cos)
    T = np.eye(4, 4)
    for i, item in enumerate(cja):
        d = DH[f'displacement_theta_{i + 1}']
        T = T.dot(
            [[cos(cja[i] + d), -sin(cja[i] + d) * cos(DH[f'alpha_{i + 1}']),
              sin(cja[i] + d) * sin(DH[f'alpha_{i + 1}']),
              DH[f'a_{i + 1}'] * cos(cja[i] + d)],
             [sin(cja[i] + d), cos(cja[i] + d) * cos(DH[f'alpha_{i + 1}']),
              -cos(cja[i] + d) * sin(DH[f'alpha_{i + 1}']),
              DH[f'a_{i + 1}'] * sin(cja[i] + d)],
             [0, sin(DH[f'alpha_{i + 1}']), cos(DH[f'alpha_{i + 1}']), DH[f'd_{i + 1}']],
             [0, 0, 0, 1]])
    return np.array(T)


def show_ijk(ax, matrix: ndarray) -> None:
    """
    Функция строит вектора i, j, k в глобальной системе координат на основе матрицы преобразования
    :param ax: Объект Axes из matplotlib
    :type ax: matplotlib.axes.Axes
    :param matrix: Матрица преобразования 4x4
    :type matrix: ndarray
    :return: None
    """
    ijk = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    rotate = ijk.dot(matrix)
    n = rotate[0:3, 3]
    ax.text(n[0], n[1], n[2], f"{n[0], n[1], n[2]}", color='red')
    colors = ['#fa0707', '#25c710', '#132ceb']
    for i, vector in enumerate(rotate):
        if i == 3:
            break
        ax.quiver(n[0], n[1], n[2], vector[0], vector[1], vector[2], color=colors[i])


def angles_from_vector(curve: Curve5x):
    """
    Функция вычисляет углы поворота для поворотного стола на основе пятиосевой траектории
    :param curve: Объект, хранящий пятиосевую траекторию
    :type curve: Curve5x
    :return: float, float
    """
    alpha = np.arctan2(curve[7], curve[6])
    beta = np.arctan2(curve[4], curve[5])
    return alpha, beta


def trajectory_generate(h: float = 10, line_width: float = 1):
    """
    Функция создает квадратную спираль
    :param h: Высота
    :type h: float
    :param line_width: Ширина линии
    :type line_width: float
    """
    v = np.array([])
    count = range(int(h / line_width))
    for i in count:
        j = i * line_width
        v = np.hstack([v, j, j, -j, -j])
        if i == np.shape(count)[0] - 1:
            v = np.hstack([v, -j, j])
    x = v[2:-1]
    y = v[3:]
    return np.vstack([x, y, np.zeros(np.shape(x)[0])])


def line_segments_array_create_from_points(points) -> ndarray:
    """
    Функция создает массив отрезков из массива точек
    :param points: Массив из точек
    :type points: ndarray
    :return: ndarray[Line_segment]
    """
    arr = np.array([])
    for i, point in enumerate(points):
        if i == np.shape(points)[0] - 1:
            break
        arr = np.hstack([arr, Line_segment(point1=point, point2=points[i + 1])])
    arr = arr[1:np.shape(arr)[0]]
    return arr


def trajectories_intersection_create(polygon: Polygon, trajectories: np.ndarray[Line_segment]) -> np.ndarray[Line_segment]:
    """
    Создание отрезков на пересечении траектории с многоугольником
    :param polygon: Многоугольник
    :type polygon: Polygon
    :param trajectories:
    :type trajectories: ndarray[Line_segment]
    :return: ndarray[Line_segment]
    """
    segments = np.array([])
    for i, item in enumerate(trajectories):
        p1 = np.array([])
        for segment in polygon.get_line_segments():
            p = point_from_segment_segment_intersection(item, segment)
            if isinstance(p, ndarray):
                if np.shape(p1) == (0,):
                    p1 = np.hstack([p1, p])
                else:
                    p1 = np.vstack([p1, p])
        if not np.shape(p1) == (3,):
            if p1.shape == (2, 3):
                s = Line_segment(point1=p1[0], point2=p1[1])
                s.color = 'red'
                s.linewidth = 6
                segments = np.hstack([segments, s])
    return segments


def cut_curve(points: ndarray, path: str = './file.stl') -> ndarray:
    """
    Функция фильтрует точки вне STL модели
    :param points: массив с точками
    :type points: ndarray
    :param path: путь до STL файла
    :type path: str
    :return: ndarray
    """
    import trimesh
    from .curve import Curve
    your_mesh = trimesh.load_mesh(path)
    arr = np.array([])
    var_mem = False
    for i, point in enumerate(points):
        var = your_mesh.contains([point])
        if var and var_mem:
            arr[-1].union(point)
        elif var and not var_mem:
            new_curve = Curve()
            new_curve.union(point)
            arr = np.hstack([arr, new_curve])
        var_mem = var
    return arr

def angle_from_vectors(vector1: ndarray, vector2: ndarray) -> ndarray:
    """
    Функция возвращает угол между двумя векторами в радианах, заданными в n_мерном пространстве
    :param vector1: Первый n-мерный вектор-строка
    :type vector1: ndarray
    :param vector2: Второй n-мерный вектор-строка
    :type vector2: ndarray
    :return: ndarray
    """
    if null_vector(vector1) or null_vector(vector2):
        raise Exception("Вектора не не могут быть равны нулю")
    cos = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return np.arccos(cos)
