import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import unittest
from unittest.mock import patch
import numpy as np
import matplotlib.pyplot as plt
import plots
import metrics


class PlotChecks(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_raw_and_smoothed_values(self, show):
        errors = np.array([.1, .08, .05, .03, .02, .01])
        fig, axes = plots.plot_stationary_errors([errors], ['Test'], 1, 2)
        np.testing.assert_array_equal(axes[0].lines[0].get_ydata(), errors)
        np.testing.assert_allclose(axes[1].lines[0].get_ydata(), metrics.trailing_mean(errors, 2), equal_nan=True)

    @patch('matplotlib.pyplot.show')
    def test_abrupt_smoothing_does_not_mix_regimes(self, show):
        errors = np.array([.01] * 6 + [.5, .4, .3, .2, .1, .01])
        fig, axes = plots.plot_abrupt_errors([errors], ['Test'], 6, 2, .05, 2)
        expected = np.r_[metrics.trailing_mean(errors[:6], 2), metrics.trailing_mean(errors[6:], 2)]
        line = next(line for line in axes[1].lines if line.get_label() == 'Test')
        np.testing.assert_allclose(line.get_ydata(), expected, equal_nan=True)

    @patch('matplotlib.pyplot.show')
    def test_repeated_seed_average(self, show):
        runs = np.array([[np.nan, .1, .2], [np.nan, .3, .6]])
        fig, ax = plots.plot_repeated_errors([runs], ['Test'], 'Test', start=1)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [.2, .4])
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [2, 3])
        vertices = ax.collections[0].get_paths()[0].vertices
        self.assertTrue(np.isfinite(vertices).all())



    @patch('matplotlib.pyplot.show')
    def test_drift_uses_actual_endpoints_without_sampling(self, show):
        theta = np.array([.2, -.7, .8])
        X = np.array([[1., 2.], [2., 1.], [-1., .5]])
        with patch('numpy.random.normal', side_effect=AssertionError('Plot must not sample a new angle')):
            fig, ax = plots.plot_drift_data(X, theta)
        for line, angle in zip(ax.lines[:2], [theta[0], theta[-1]]):
            np.testing.assert_allclose(line.get_xdata(), [-8 * np.cos(angle), 8 * np.cos(angle)])
            np.testing.assert_allclose(line.get_ydata(), [-8 * np.sin(angle), 8 * np.sin(angle)])
        np.testing.assert_allclose(ax.collections[0].get_offsets(), X)

    @patch('matplotlib.pyplot.show')
    def test_cumulative_plot_starts_at_observation_one(self, show):
        # the notebooks accumulate the raw errors from observation 1 (np.cumsum(errors)),
        # as declared in the benchmark CSV: entry i is plotted at observation i + 1
        errors = np.array([.5, .2, .1, .05])
        totals = np.cumsum(errors)
        fig, ax = plots.plot_cumulative_error([totals], ['Test'], 'Test')
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [.5, .7, .8, .85])
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 2, 3, 4])

    @patch('matplotlib.pyplot.show')
    def test_captured_variance_curve_and_optimal_reference(self, show):
        E = np.array([0., .25, 1.])
        percentages = 100 * (5 - 4.5 * E) / 5.5
        best = 100 * 5 / 5.5
        fig, ax = plots.plot_captured_variance([percentages], ['Test'], 'Test', best)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), percentages)
        np.testing.assert_allclose(ax.lines[1].get_ydata(), best)

    @patch('matplotlib.pyplot.show')
    def test_repeated_band_matches_percentiles(self, show):
        runs = np.array([[.1, .2, .3], [.2, .5, .8], [.6, .7, .9]])
        fig, ax = plots.plot_repeated_errors([runs], ['Test'], 'Test', warmup=1, upper_limit=.95)
        low, high = np.percentile(runs, [10, 90], axis=0)
        vertices = ax.collections[0].get_paths()[0].vertices
        for step, lo, hi in zip([1, 2, 3], low, high):
            y = vertices[np.isclose(vertices[:, 0], step), 1]
            self.assertAlmostEqual(y.min(), lo)
            self.assertAlmostEqual(y.max(), hi)
        np.testing.assert_allclose(ax.get_xlim(), [2, 3])
        np.testing.assert_allclose(ax.get_ylim(), [0, .95])


if __name__ == '__main__':
    unittest.main()
