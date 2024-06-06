from __future__ import annotations
import numpy as np
from math import acos
# from .line import Line

def angle_from_vectors_2d(v1: list | np.ndarray, v2: list | np.ndarray) -> np.ndarray:
    """
    Функция возвращает угол между двумя векторами
    :param v1: Вектор 1
    :type v1: np.ndarray
    :param v2: Вектор 2
    :type v2: np.ndarray
    :return: np.ndarray
    """
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    matrix = np.array([v1, v2])
    det = np.linalg.det(matrix)
    if det == 0:
        if opposite_vectors(v1, v2):
            alpha = np.pi
        else:
            alpha = 0
    else:
        alpha = det / abs(det) * acos(cos)
    return alpha

def opposite_vectors(v1, v2) -> bool:
    """
    Функция проверяет, противоположны ли векторы
    :param v1: Вектор 1
    :type v1: np.ndarray
    :param v2: Вектор 2
    :type v2: np.ndarray
    :return: bool
    """
    v1 = np.array(v1)/np.linalg.norm(v1)
    v2 = np.array(v2) / np.linalg.norm(v2)
    if np.sum(v1+v2) == 0.0:
        return True
    else:
        return False
def vector_rotation(vector, angle, grad=False) -> np.ndarray:
    """
    Вращение вектора по часовой стрелке
    :param vector: Координаты вектора размерностью 2
    :type vector: np.ndarray
    :param angle: Угол в радианах
    :type angle: float
    :param grad: В градусах ли даны углы, если да, то в angle писать градусы
    :return: np.ndarray
    """
    if grad:
        angle = angle*np.pi/180
    matrix_rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    new_vector = np.round(matrix_rotation.dot(vector), 6)
    return new_vector

def perpendicular_line(line: Line, left: bool = False) -> np.ndarray:
    """
    Функция возвращает линию перпендикулярную данной в 2D пространстве
    :param line: Объект линии
    :type line: Line
    :param left: Перпендикулярная линия вправо или влево
    :return: np.ndarray
    """
    vector = line.coeffs()[3:5]
    if left:
        angle = -90
    else:
        angle = 90
    new_vector = vector_rotation(vector, angle, grad=True)
    new_line = Line(line.a, line.b, line.c, new_vector[0], new_vector[1], 0)
    return new_line
