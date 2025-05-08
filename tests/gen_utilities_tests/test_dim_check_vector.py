import unittest

import numpy as np

from rock_physics_open.equinor_utilities.gen_utilities import dim_check_vector


class DimCheckVectorTestCase(unittest.TestCase):
    def test_dim_check_vector_ok(self):
        a = np.ones(11)
        b = 42
        c = True
        a_ref = np.ones(11)
        b_ref = np.ones_like(a) * 42
        c_ref = np.ones_like(a).astype(bool)
        a_out, b_out, c_out = dim_check_vector((a, b, c))
        np.testing.assert_equal(a_out, a_ref)
        np.testing.assert_equal(b_out, b_ref)
        np.testing.assert_equal(c_out, c_ref)

    def test_dim_check_vector_force(self):
        a = np.ones(11)
        b = 42
        c = True
        a_ref = np.ones(11)
        b_ref = (np.ones_like(a) * 42).astype(int)
        c_ref = np.ones_like(a).astype(bool)
        a_out, b_out, c_out = dim_check_vector((a, b, c), force_type=np.dtype(float))
        np.testing.assert_equal(a_out, a_ref)
        np.testing.assert_equal(b_out, b_ref)
        np.testing.assert_equal(c_out, c_ref)
        assert a_out.dtype == a_ref.dtype
        assert b_out.dtype != b_ref.dtype
        assert c_out.dtype != c_ref.dtype


if __name__ == "__main__":
    unittest.main()
