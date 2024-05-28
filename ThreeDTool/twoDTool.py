import numpy as np
from math import acos
from loguru import logger
# from plane import Plane


def angle_from_vectors_2d(v1, v2):
    """
    Функция возвращает
    :param v1:
    :param v2:
    :return:
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

def opposite_vectors(v1, v2):
    """
    Функция проверяет, противоположны ли векторы
    :param v1:
    :param v2:
    :return:
    """
    v1 = np.array(v1)/np.linalg.norm(v1)
    v2 = np.array(v2) / np.linalg.norm(v2)
    if np.sum(v1+v2) == 0.0:
        return True
    else:
        return False
def vector_rotation(vector, angle, grad=False):
    """
    Вращение вектора по часовой стрелке
    :param vector:
    :param angle:
    :param grad:
    :return:
    """
    if grad:
        angle = angle*np.pi/180
    matrix_rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    new_vector = np.round(matrix_rotation.dot(vector), 6)
    return new_vector
