import unittest

from rock_physics_open.equinor_utilities.gen_utilities import dict_value_to_float


class DictToFloatTestCase(unittest.TestCase):
    def test_dict_to_float(self):
        inp_dict = {"a": "1.0", "b": "2", "d": "[1.2, 4.5]", "e": "-3"}
        ref_dict = {"a": 1.0, "b": 2.0, "d": [1.2, 4.5], "e": -3.0}
        out_dict = dict_value_to_float(inp_dict)
        assert out_dict == ref_dict


if __name__ == "__main__":
    unittest.main()
