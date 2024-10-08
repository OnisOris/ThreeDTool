from ..polygon import Polygon
import numpy as np


class TestPolygon:
    # polygons_equal ---------------------------
    def test_polygons_equal(self):
        """
        Тест на сдвинутые вершины (1 против часовой, 2 с отставанием на одну вершину)
        """
        pol1 = Polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        pol2 = Polygon([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])
        assert pol1.polygons_equal(pol2)

    def test_polygons_equal2(self):
        """
        Тест на сдвинутые вершины (1 против часовой, 2 по часовой и со сдвигом на одну вершину)
        """
        pol1 = Polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        pol2 = Polygon([[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]])
        assert pol1.polygons_equal(pol2)

    def test_polygons_equal3(self):
        """
        Тест на точность. В функции максимальная точность 10^-6: вершины в этом диапазоне будут одинаковыми.
        """
        pol1 = Polygon([[0, 0, 0.000000004], [1, -0.000000003, 0], [1, 1, 0], [0, 1, 0]])
        pol2 = Polygon([[1, 0.000000003, 0], [0, 0, 0], [0, 1, 0], [1.000000004, 1, 0]])
        assert pol1.polygons_equal(pol2)

    # incorrect
    def test_polygons_equal4(self):
        """
        Тест на неравенство многоугольников
        """
        pol1 = Polygon([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
        pol2 = Polygon([[1.5, 0, 0], [1.5, 1, 0], [0.5, 1, 0], [0.5, 0, 0]])
        assert not pol1.polygons_equal(pol2)

    def test_polygons_equal5(self):
        """
        Тест на неравенство многоугольников
        """
        pol1 = Polygon([[0, 0, 0], [1, 0.00000004, 0], [1, 1, 0], [0, 1, 0]])
        pol2 = Polygon([[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]])
        assert not pol1.polygons_equal(pol2)

    def test_polygons_equal6(self):
        pol1 = Polygon([[0, 0, 0.], [1, -0.000000003, 0], [1, 1, 0], [0, 1, 0]])
        pol2 = Polygon([[1, 0.000000009, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]])
        assert not pol1.polygons_equal(pol2)

    # points_from_polygon_polygon_intersection_and_rot_v ---------------------------
    def test_points_from_polygon_polygon_intersection_and_rot_v(self):
        from ..threeDTool import rot_v
        pol1 = Polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        points_pol = np.array([[1.5, 0, 0], [1.5, 1, 0], [0.5, 1, 0], [0.5, 0, 0]])
        axis = [1, 1, 1]
        points_pol = rot_v(points_pol, np.pi / 6, axis)
        pol2 = Polygon(points_pol)
        points_test = np.array([[1., 0.73205081, 0.], [0.6830127, 0.5, 0.]])
        assert np.allclose(pol1.points_from_polygon_polygon_intersection(pol2), points_test, 1e-8)
