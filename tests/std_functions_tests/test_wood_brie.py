from typing import final

import numpy as np
from numpy.random import default_rng

from rock_physics_open.equinor_utilities.std_functions import brie, multi_wood, wood


class TestWoodBrie:
    def setup(self):
        s_gas = np.array([1.0, 0.0, 0.0])
        s_brine = np.array([0.0, 1.0, 0.0])
        s_oil = np.array([0.0, 0.0, 1.0])
        k_gas = np.array([0.25e6, 0.25e6, 0.25e6])
        k_brine = np.array([2.8e9, 2.8e9, 2.8e9])
        k_oil = np.array([0.9e9, 0.9e9, 0.9e9])
        rho_gas = np.array([210, 210, 210])
        rho_brine = np.array([1005, 1005, 1005])

        return s_gas, s_brine, s_oil, k_gas, k_brine, k_oil, rho_gas, rho_brine

    def test_wood(self):
        s_gas, _, _, k_gas, k_brine, _, rho_gas, rho_brine = self.setup()
        k, rho = wood(s_gas, k_gas, rho_gas, k_brine, rho_brine)
        k_ref = np.array([0.25e6, 2.8e9, 2.8e9])
        rho_ref = np.array([210, 1005, 1005])
        np.testing.assert_almost_equal(k, k_ref)
        np.testing.assert_almost_equal(rho, rho_ref)

    def test_brie(self):
        s_gas, s_brine, s_oil, k_gas, k_brine, k_oil, _, _ = self.setup()
        e = 1.5
        k = brie(s_gas, k_gas, s_brine, k_brine, s_oil, k_oil, e)
        k_ref = np.array([0.25e6, 2.8e9, 0.9e9])
        np.testing.assert_almost_equal(k, k_ref)

    def test_wood_brie_mix(self):
        rg = default_rng(12345)
        s_gas = 0.5 * rg.random(11)
        s_oil = 0.5 * rg.random(11)
        s_brine = 1.0 - (s_gas + s_oil)
        k_gas = 0.25e6 * np.ones_like(s_gas)
        k_brine = 2.8e9 * np.ones_like(s_gas)
        k_oil = 0.9e9 * np.ones_like(s_gas)
        e = 1.7
        k_w = wood(s_gas, k_gas, np.ones_like(s_gas), k_brine, np.ones_like(s_gas))[0]
        k_b = brie(s_gas, k_gas, s_brine, k_brine, s_oil, k_oil, e)
        k_w_ref = np.array(
            [
                2197857.3032863,
                1577741.8284602,
                626980.6039708,
                739237.2282665,
                1277944.8143517,
                1501669.3287616,
                835514.1581449,
                2675283.4430605,
                743080.5661045,
                530843.4103878,
                2012865.2583938,
            ]
        )
        k_b_ref = np.array(
            [
                1.2081004,
                1.3510999,
                1.6787997,
                1.2305812,
                1.111577,
                1.3090595,
                1.4636728,
                1.4097729,
                1.5565275,
                1.5423466,
                2.1396782,
            ]
        )
        np.testing.assert_almost_equal(k_w, k_w_ref)
        np.testing.assert_almost_equal(k_b / 1e9, k_b_ref)


@final
class TestMultiWood:
    s_gi = 0.0
    s_oi = 0.5
    s_wi = 0.5
    oil_init_k = 492_032_461.8739312
    gas_init_k = 208_287_817.8755032
    brine_init_k = 2_814_732_817.889945
    expected_value = 837_640_292.3902907

    def test_multi_wood_array(self):
        """Test multi_wood with numpy array inputs."""
        result = multi_wood(
            fractions=[np.array(self.s_oi), np.array(self.s_gi), np.array(self.s_wi)],
            bulk_moduli=[
                np.array(self.oil_init_k),
                np.array(self.gas_init_k),
                np.array(self.brine_init_k),
            ],
        )
        expected = np.array([self.expected_value])
        np.testing.assert_almost_equal(result, expected)

    def test_multi_wood_scalar(self):
        """Test multi_wood with scalar inputs."""
        result = multi_wood(
            fractions=[self.s_oi, self.s_gi, self.s_wi],
            bulk_moduli=[self.oil_init_k, self.gas_init_k, self.brine_init_k],
        )
        expected = self.expected_value
        np.testing.assert_almost_equal(result, expected)
