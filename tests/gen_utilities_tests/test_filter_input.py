import unittest

import numpy as np

from rock_physics_open.equinor_utilities.gen_utilities import filter_input_log


class FilterInputTestCase(unittest.TestCase):
    def test_filter_input(self):
        a = np.ones(11).astype(float)
        a[4] = np.nan
        b = np.zeros_like(a)
        b[1] = 3.14
        b[9] = np.inf
        b[10] = -np.inf
        idx_ref = np.ones_like(a).astype(bool)
        idx_ref[4] = False
        idx_ref[9:] = False
        a_ref = a[idx_ref]
        b_ref = b[idx_ref]
        idx, (a_out, b_out) = filter_input_log((a, b))
        np.testing.assert_equal(idx, idx_ref)
        np.testing.assert_equal(a_out, a_ref)
        np.testing.assert_equal(b_out, b_ref)

    def test_filter_input_negative(self):
        a = np.ones(11).astype(float)
        b = np.zeros_like(a)
        b[4:6] = -1.0
        idx_ref = np.ones_like(a).astype(bool)
        idx_ref[4:6] = False
        a_ref = a[idx_ref]
        b_ref = b[idx_ref]
        idx, (a_out, b_out) = filter_input_log((a, b))
        np.testing.assert_equal(idx, idx_ref)
        np.testing.assert_equal(a_out, a_ref)
        np.testing.assert_equal(b_out, b_ref)

    def test_filter_input_no_zero(self):
        a = np.ones(11).astype(float)
        b = np.zeros_like(a)
        b[4:6] = 1.0
        idx_ref = np.zeros_like(a).astype(bool)
        idx_ref[4:6] = True
        a_ref = a[idx_ref]
        b_ref = b[idx_ref]
        idx, (a_out, b_out) = filter_input_log((a, b), no_zero=True)
        np.testing.assert_equal(idx, idx_ref)
        np.testing.assert_equal(a_out, a_ref)
        np.testing.assert_equal(b_out, b_ref)

    def test_filter_input_positive(self):
        a = np.ones(11).astype(float)
        b = np.zeros_like(a)
        b[4:6] = -1.0
        idx_ref = np.zeros_like(a).astype(bool)
        idx_ref[4:6] = True
        a_ref = a[idx_ref]
        b_ref = b[idx_ref]
        idx, (a_out, b_out) = filter_input_log((a, b), negative=True, positive=False)
        np.testing.assert_equal(idx, idx_ref)
        np.testing.assert_equal(a_out, a_ref)
        np.testing.assert_equal(b_out, b_ref)

    def test_filter_input_bool(self):
        a = np.ones(11).astype(float)
        b = np.ones_like(a).astype(bool)
        b[4:6] = False
        idx_ref = np.ones_like(a).astype(bool)
        idx_ref[4:6] = False
        a_ref = a[idx_ref]
        b_ref = b[idx_ref]
        idx, (a_out, b_out) = filter_input_log((a, b))
        np.testing.assert_equal(idx, idx_ref)
        np.testing.assert_equal(a_out, a_ref)
        np.testing.assert_equal(b_out, b_ref)

    def test_filter_input_str(self):
        a = np.ones(11).astype(float)
        a[4:6] = -1.0
        b = np.array(["NaN"] * 11)
        idx_ref = np.ones_like(a).astype(bool)
        idx_ref[4:6] = False
        a_ref = a[idx_ref]
        b_ref = b[idx_ref]
        idx, (a_out, b_out) = filter_input_log((a, b))
        np.testing.assert_equal(idx, idx_ref)
        np.testing.assert_equal(a_out, a_ref)
        np.testing.assert_equal(b_out, b_ref)


if __name__ == "__main__":
    unittest.main()
