"""Checks of the Part 1 block functions shared by 02_sliding_window and 03_oja
(benchmark.run_method / single_run / repeat_runs / best_time, metrics.recovery_delays, plots.show_*)."""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import matplotlib.pyplot as plt

import benchmark
import metrics
import plots
from data_generators import generate_stationary, generate_abrupt
from methods import forgetting_pca, oja_pca


SEEDS = range(3)
N = 400


def make_streams(generator, **kwargs):
    return {s: generator(s, n_samples=N, **kwargs) for s in SEEDS}


def settings(csv_path, abrupt=False):
    return {'n_samples': N, 'seeds': SEEDS, 'lam1': 5.0, 'lam2': 0.5, 'theta0': 0.0,
            'theta1': np.pi / 3 if abrupt else '', 'switch_step': 200 if abrupt else '',
            'warmup': 50, 'smooth_window': 20,
            'recovery_threshold': 0.05 if abrupt else '', 'recovery_hold': 20 if abrupt else '',
            'results_csv': csv_path}


class RunningChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.csv = Path(self.tmp.name) / 'results.csv'
        self.stationary = make_streams(generate_stationary)
        self.abrupt = make_streams(generate_abrupt, switch_step=200)

    def tearDown(self):
        self.tmp.cleanup()

    def test_run_method_matches_direct_computation(self):
        runs, angles, runtimes = benchmark.run_method(lambda X: forgetting_pca(X, 0.05), self.stationary)
        self.assertEqual(runs.shape, (len(SEEDS), N))
        self.assertEqual(len(runtimes), len(SEEDS))
        for row, angle_row, (X, theta, truth) in zip(runs, angles, self.stationary.values()):
            directions = forgetting_pca(X, 0.05)[0]
            np.testing.assert_array_equal(row, metrics.axis_error(directions, truth))
            np.testing.assert_array_equal(angle_row, metrics.axis_angle_error(directions, theta))

    def test_single_run_one_entry_per_value_in_order(self):
        values = [0.001, 0.01, 0.005]
        out = benchmark.single_run(oja_pca, values, self.stationary[0])
        self.assertEqual(list(out), values)
        X, _, truth = self.stationary[0]
        for value, (directions, errors) in out.items():
            np.testing.assert_array_equal(directions, oja_pca(X, value)[0])
            np.testing.assert_array_equal(errors, metrics.axis_error(directions, truth))

    def test_repeat_runs_saves_one_row_per_value_in_the_named_column(self):
        with patch('builtins.print'):
            out = benchmark.repeat_runs(forgetting_pca, [0.02, 0.1], self.stationary, 'stationary',
                                        settings(self.csv), 'Test PCA', 'alpha')
            benchmark.repeat_runs(lambda X, w: forgetting_pca(X, 2 / (w + 1)), [19], self.stationary, 'stationary',
                                  settings(self.csv), 'Test window', 'window_size')
        self.assertEqual(list(out), [0.02, 0.1])
        expected = benchmark.run_method(lambda X: forgetting_pca(X, 0.02), self.stationary)[0]
        np.testing.assert_array_equal(out[0.02][0], expected)
        rows = list(csv.DictReader(self.csv.open(encoding='utf-8')))
        self.assertEqual([(r['method'], r['alpha'], r['window_size']) for r in rows],
                         [('Test PCA', '0.02', ''), ('Test PCA', '0.1', ''), ('Test window', '', '19')])
        self.assertAlmostEqual(float(rows[0]['mean_error_after_warmup']), expected[:, 50:].mean())

    def test_recovery_delays_matches_recovery_delay(self):
        runs = benchmark.run_method(lambda X: forgetting_pca(X, 0.1), self.abrupt)[0]
        s = settings(self.csv, abrupt=True)
        expected = [metrics.recovery_delay(e, 200, threshold=0.05, window=20, hold=20) for e in runs]
        np.testing.assert_array_equal(metrics.recovery_delays(runs, s), expected)

    def test_best_time_is_the_minimum(self):
        with patch('time.perf_counter', side_effect=[0.0, 3.0, 10.0, 11.0, 20.0, 22.0]):
            self.assertEqual(benchmark.best_time(lambda: None, repeats=3), 1.0)


class DisplayChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.csv = Path(self.tmp.name) / 'results.csv'
        self.tables = []

    def tearDown(self):
        plt.close('all')
        self.tmp.cleanup()

    def capture(self, headers, rows):
        self.tables.append((headers, rows))

    def blocks(self, scenario, streams):
        s = settings(self.csv, abrupt=scenario == 'abrupt')
        values = [0.1, 0.02]
        with patch('builtins.print'):
            single = benchmark.single_run(forgetting_pca, values, streams[0])
            results = benchmark.repeat_runs(forgetting_pca, values, streams, scenario, s, 'Test PCA', 'alpha')
        return s, values, single, results

    @patch('matplotlib.pyplot.show')
    def test_tables_contain_the_expected_values(self, show):
        streams = make_streams(generate_abrupt, switch_step=200)
        s, values, single, results = self.blocks('abrupt', streams)
        label = lambda a: f'a={a:g}'
        with patch('plots.show_table', side_effect=self.capture):
            plots.show_single_run(single, 'abrupt', s, label)
            plots.show_cost(results, label)
            plots.show_repeated_measurements(results, 'abrupt', s, label)
        single_rows, cost_rows, repeated_rows = (rows for _, rows in self.tables)
        self.assertEqual([r[0] for r in single_rows], ['a=0.1', 'a=0.02'])
        errors = single[0.1][1]
        self.assertEqual(single_rows[0][2], f'{errors[200:].mean():.5f}')
        self.assertEqual(cost_rows[0][1], results[0.1][2].mean())
        runs = results[0.02][0]
        delays = metrics.recovery_delays(runs, s)
        self.assertEqual(repeated_rows[1][4], f'{np.isfinite(delays).sum()}/{len(SEEDS)}')
        self.assertEqual(repeated_rows[1][2].split(' ')[0], f'{runs[:, 200:].mean(axis=1).mean():.5f}')

    @patch('matplotlib.pyplot.show')
    def test_show_plots_draws_every_figure_for_every_regime(self, show):
        for scenario, streams in [('stationary', make_streams(generate_stationary)),
                                  ('abrupt', make_streams(generate_abrupt, switch_step=200))]:
            s, values, single, results = self.blocks(scenario, streams)
            with patch('builtins.print'):
                plots.show_plots(single, results, streams[0][1], scenario, s, 'Test', lambda a: f'a={a:g}')
        self.assertEqual(show.call_count, 2 * 5)


if __name__ == '__main__':
    unittest.main()
