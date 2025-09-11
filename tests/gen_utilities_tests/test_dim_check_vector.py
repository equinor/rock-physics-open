import numpy as np
import pandas as pd

from rock_physics_open.equinor_utilities.gen_utilities import dim_check_vector


class TestDimCheckVector:
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

    def test_dim_check_vector_single_np_array(self):
        """Test that a single numpy array is returned as is."""
        a = np.ones(11)
        a_out = dim_check_vector(a)
        np.testing.assert_equal(a_out, a)

    def test_dim_check_vector_single_np_array_force(self):
        """Test that a single numpy array is returned as is."""
        a = np.ones(11).astype(float)
        dtype = np.dtype(int)
        a_out = dim_check_vector(a, force_type=dtype)
        np.testing.assert_equal(a_out, a)
        assert a_out.dtype == dtype

    def test_dim_check_vector_single_df(self):
        """Test that a single pandas dataframe is returned as is."""
        a = np.ones((11, 3))
        a_df = pd.DataFrame(a)
        a_out = dim_check_vector(a_df)
        pd.testing.assert_frame_equal(a_out, a_df)

    def test_dim_check_vector_single_df_force(self):
        """Test that a single pandas dataframe is returned as is."""
        a = np.ones((11, 3))
        a_df = pd.DataFrame(a).astype(float)
        dtype = np.dtype(int)
        a_out = dim_check_vector(a_df, force_type=dtype)
        pd.testing.assert_frame_equal(left=a_out, right=a_df, check_dtype=False)
        assert all(a_out.dtypes == dtype)

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
        if isinstance(a_out, np.ndarray):
            assert a_out.dtype == a_ref.dtype
        if isinstance(b_out, np.ndarray):
            assert b_out.dtype != b_ref.dtype
        if isinstance(c_out, np.ndarray):
            assert c_out.dtype != c_ref.dtype
