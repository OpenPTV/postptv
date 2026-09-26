import unittest

import numpy as np

from flowtracks import interpolation


class TestIDWCall(unittest.TestCase):
    def setUp(self):
        # Simple 2D case for easy checking
        self.tracer_pos = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        self.data = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ])
        self.interp_points = np.array([
            [0.5, 0.5, 0.0],
        ])

    def test_basic_idw(self):
        idw = interpolation.InverseDistanceWeighter(num_neighbs=4, param=1)
        result = idw(self.tracer_pos, self.interp_points, self.data)
        self.assertEqual(result.shape, (1, 3))
        # Should be a weighted average, check sum is reasonable
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))

    def test_empty_tracers(self):
        import warnings
        idw = interpolation.InverseDistanceWeighter(num_neighbs=2, param=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = idw(np.empty((0, 3)), self.interp_points, np.empty((0, 3)))
            self.assertTrue(any(issubclass(warn.category, UserWarning) for warn in w))
        self.assertEqual(result.shape, (1, 3))
        self.assertTrue(np.all(result == 0))

    def test_1d_data(self):
        data_1d = np.array([1.0, 2.0, 3.0, 4.0])
        idw = interpolation.InverseDistanceWeighter(num_neighbs=4, param=1)
        result = idw(self.tracer_pos, self.interp_points, data_1d)
        self.assertEqual(result.shape, (1, 1))
        self.assertTrue(np.all(result >= 1))
        self.assertTrue(np.all(result <= 4))

    def test_companionship(self):
        # Exclude the first tracer from the interpolation
        import warnings
        companions = np.array([0])
        idw = interpolation.InverseDistanceWeighter(num_neighbs=4, param=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = idw(self.tracer_pos, self.interp_points, self.data, companionship=companions)
            self.assertTrue(any(issubclass(warn.category, RuntimeWarning) for warn in w))
        self.assertEqual(result.shape, (1, 3))
        # The result should not be equal to the average including the first tracer
        idw_all = interpolation.InverseDistanceWeighter(num_neighbs=4, param=1)
        result_all = idw_all(self.tracer_pos, self.interp_points, self.data)
        self.assertFalse(np.allclose(result, result_all))
