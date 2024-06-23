from ..surface import Sphere
import random
import numpy as np


class TestSphere:
    def test_init(self):
        """
        Проверка уравнения (x-a)^2 + (y - b)^2 + (z - c)^2 = r^2. Сфера генерируется в произвольной точке от 0 до 1000
         по x, y, z и по z тестовая точка проверяется по нескольким точкам поверхности сферы
        """
        xyz = np.array([random.random() * 1000, random.random() * 1000, random.random() * 1000, random.random() * 100])
        sp = Sphere(*xyz)
        res_point1 = xyz[0:3] + np.array([0, 0, xyz[3]])
        res_point2 = xyz[0:3] + np.array([0, xyz[3], 0])
        res_point3 = xyz[0:3] + np.array([xyz[3], 0, 0])

        assert (round((res_point1[0] - sp.a) ** 2 + (res_point1[1] - sp.b) ** 2 + (res_point1[2] - sp.c) ** 2, 5)
                == round(sp.r ** 2, 5)
                and round((res_point2[0] - sp.a) ** 2 + (res_point2[1] - sp.b) ** 2 + (res_point2[2] - sp.c) ** 2, 5)
                == round(sp.r ** 2, 5)
                and round((res_point3[0] - sp.a) ** 2 + (res_point3[1] - sp.b) ** 2 + (res_point3[2] - sp.c) ** 2, 5)
                == round(sp.r ** 2, 5)
                )



