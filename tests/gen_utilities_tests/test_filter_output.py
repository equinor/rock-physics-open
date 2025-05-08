import unittest

import numpy as np

from rock_physics_open.equinor_utilities.gen_utilities import filter_output


class FilterOutputTestCase(unittest.TestCase):
    def test_filter_output(self):
        a_inp = np.linspace(0, 1, 11)
        b_inp = np.linspace(0, 1, 11) * 2
        idx = np.ones(15).astype(bool)
        idx[6:8] = False
        idx[12:14] = False
        a_ref = np.ones_like(idx) * np.nan
        a_ref[idx] = a_inp
        b_ref = np.ones_like(idx) * np.nan
        b_ref[idx] = b_inp
        a_out, b_out = filter_output(idx, (a_inp, b_inp))
        np.testing.assert_equal(a_out, a_ref)
        np.testing.assert_equal(b_out, b_ref)


if __name__ == "__main__":
    unittest.main()
