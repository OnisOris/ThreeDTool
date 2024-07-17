import numpy as np

from ..up import Up

class TestUp:
    def test_add(self):
        up1 = Up([1, 0, 0])
        up2 = Up([2, 2, 2])
        up3 = Up([3, 2, 2])
        assert np.allclose(up1 + up2, up3, 1e-8)
    def test_add2(self):
        up1 = Up([[1, 0, 0]])
        up2 = Up([2, 2, 2])
        up3 = Up([3, 2, 2])
        assert np.allclose(up1 + up2, up3, 1e-8)

