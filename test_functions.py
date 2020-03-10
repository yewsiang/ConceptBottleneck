
import unittest
import analysis
import numpy as np

def assert_np_arrays_similar(arr1, arr2, tol):
    """Checks elementwise if two numpy arrays are within tolerance.

    If there are nans in the array, then both arrays must contain nans.
    Throws assertion error if arrays are not similar.

    Args:
      arr1 (float): N-dims tensor
      arr2 (float): N-dims tensor
      tol  (float): Tolerance. Value of the elementwise difference between
                    arr1 and arr2 that can be tolerated.

    """
    # Check shapes
    if arr1.shape != arr2.shape: assert False, 'Shapes should be the same.'

    # Check nans
    xor = np.logical_xor(np.isnan(arr1), np.isnan(arr2))
    # xor should all be False if nans correspond to the same places
    # Hence, the sum of all xors should be 0
    if np.sum(xor) != 0: assert False, 'nans do not coincide.'

    # Look at differences of elements that are not nans
    diff = arr1 - arr2
    not_nan = np.logical_not(np.isnan(diff))
    max_diff = np.max(np.abs(diff[not_nan]))
    if max_diff > tol: assert False, 'Max diff value exceeds tolerance'

class TestAnalysisMethods(unittest.TestCase):
    def test_assign_value_to_bins(self):
        bins = np.array([0., 1., 1.3, 2., 3.5, 5.])
        values = np.array([-10., 0., 0.4, 0.6, 1.4, 2.5, 3.0, 4.5, 10.])
        expected = np.array([0., 0., 0., 1., 2., 3., 4., 5., 5.])
        out = analysis.assign_value_to_bins(values, bins, use_integer_bins=True)
        assert_np_arrays_similar(out, expected, tol=1e-5)

        bins = np.array([0., 1., 1.3, 2., 3.5, 5.])
        values = np.array([-10., 0., 0.4, 0.6, 1.4, 2.5, 3.0, 4.5, 10.])
        expected = np.array([0., 0., 0., 1., 1.3, 2., 3.5, 5., 5.])
        out = analysis.assign_value_to_bins(values, bins, use_integer_bins=False)
        assert_np_arrays_similar(out, expected, tol=1e-5)
        true = np.array([0., 0., 5., 3.5, 3.5, 2., 1.3, 1.3, 1.])
        out1, out2 = analysis.convert_continuous_back_to_ordinal(true, values, use_integer_bins=False)
        exp1 = expected
        exp2 = np.array([0., 0., 5., 4., 4., 3., 2., 2., 1.])
        assert_np_arrays_similar(out1, exp1, tol=1e-5)
        assert_np_arrays_similar(out2, exp2, tol=1e-5)
        out3, out4 = analysis.convert_continuous_back_to_ordinal(true, values, use_integer_bins=True)
        exp3 = np.array([0., 0., 0., 1., 2., 3., 4., 5., 5.])
        exp4 = exp2
        assert_np_arrays_similar(out3, exp3, tol=1e-5)
        assert_np_arrays_similar(out4, exp4, tol=1e-5)

        bins = np.array([0., 1., 1.3, 2., 3.5, 5.])
        values = np.array([[-10., 0., 0.4, 0.6, 1.4, 2.5, 3.0, 4.5, 10.], [-10., 0., 0.4, 0.6, 1.4, 2.5, 3.0, 4.5, 10.]])
        expected = np.array([[0., 0., 0., 1., 1.3, 2., 3.5, 5., 5.], [0., 0., 0., 1., 1.3, 2., 3.5, 5., 5.]])
        out = analysis.assign_value_to_bins(values, bins, use_integer_bins=False)
        assert_np_arrays_similar(out, expected, tol=1e-5)



if __name__ == '__main__':
    unittest.main()