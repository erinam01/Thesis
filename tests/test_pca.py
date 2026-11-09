from pathlib import Path
import csv
import tempfile
import unittest
import numpy as np

import data_generators as data
import methods
import metrics
import benchmark


class MigrationRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = np.load(Path(__file__).with_name('reference_outputs.npz'))

    def test_generators_match_original(self):
        for scenario in ('stationary', 'abrupt'):
            fn = getattr(data, 'generate_' + scenario)
            options = dict(n_samples=400, lam1=5.0, lam2=0.5, theta0=0.0)
            if scenario == 'abrupt':
                options.update(theta1=np.pi / 3, switch_step=250)
            for name, array in zip(('X', 'theta', 'truth'), fn(13, **options)):
                np.testing.assert_array_equal(array, self.reference[scenario + '_' + name])

    def test_methods_match_original(self):
        cases = [('cumulative_pca', ()), ('forgetting_pca', (0.02,)),
                 ('sliding_window_pca', (80,)), ('adaptive_gain_pca', ())]
        for scenario in ('stationary', 'abrupt'):
            X = self.reference[scenario + '_X'].copy()
            X.setflags(write=False)
            for name, args in cases:
                with self.subTest(scenario=scenario, method=name):
                    result = getattr(methods, name)(X, *args)
                    key = scenario + '_' + name
                    q, expected = result[0], self.reference[key + '_q']
                    np.testing.assert_array_equal(np.isnan(q), np.isnan(expected))
                    valid = np.isfinite(q).all(axis=1)
                    np.testing.assert_allclose(np.abs(np.sum(q[valid] * expected[valid], axis=1)), 1, atol=1e-10)
                    if isinstance(result[1], np.ndarray):
                        np.testing.assert_allclose(result[1], self.reference[key + '_C'], atol=1e-12)
                    else:
                        np.testing.assert_allclose(result[1].mean_, self.reference[key + '_mean'], atol=1e-12)
                    if len(result) == 3:
                        np.testing.assert_allclose(result[2], self.reference[key + '_gains'], atol=1e-12)

    def test_recursive_covariance_and_window(self):
        X = self.reference['stationary_X'][:100]
        q, C = methods.cumulative_pca(X)
        np.testing.assert_allclose(C, X.T @ X / len(X), atol=1e-12)
        wq, wC = methods.sliding_window_pca(X, len(X))
        np.testing.assert_allclose(C, wC, atol=1e-12)
        np.testing.assert_allclose(metrics.axis_error(q, wq), 0, atol=1e-12)
        _, C = methods.sliding_window_pca(X, 20)
        np.testing.assert_allclose(C, X[-20:].T @ X[-20:] / 20, atol=1e-12)

    def test_gain_conventions_and_causality(self):
        X = self.reference['abrupt_X']
        directions, covariance, gains = methods.adaptive_gain_pca(X)
        prefix, _, prefix_gains = methods.adaptive_gain_pca(X[:300])
        np.testing.assert_allclose(prefix, directions[:300], atol=1e-12)
        np.testing.assert_allclose(prefix_gains, gains[:300], atol=1e-12)
        self.assertTrue(np.all(gains >= .005 - 1e-14))
        self.assertTrue(np.all(gains <= .05 + 1e-14))
        np.testing.assert_allclose(gains[:200], .02, atol=1e-14)

        # The recorded gains must reconstruct the returned covariance.
        expected = np.zeros((2, 2))
        for x, alpha in zip(X, gains):
            expected = (1 - alpha) * expected + alpha * np.outer(x, x)
        np.testing.assert_allclose(covariance, expected, atol=1e-12)


    def test_explicit_generator_settings(self):
        X, theta, truth = data.generate_abrupt(7, n_samples=80, lam1=2, lam2=.1,
                                               theta0=.2, theta1=.8, switch_step=30)
        self.assertEqual(X.shape, (80, 2))
        np.testing.assert_allclose(theta[:30], .2)
        np.testing.assert_allclose(theta[30:], .8)
        np.testing.assert_allclose(np.linalg.norm(truth, axis=1), 1)

    def test_metrics_and_recovery(self):
        truth = self.reference['stationary_truth']
        np.testing.assert_allclose(metrics.axis_error(truth, -truth), 0, atol=1e-12)
        np.testing.assert_allclose(metrics.axis_angle_error(-truth, self.reference['stationary_theta']), 0, atol=1e-12)
        self.assertEqual(metrics.recovery_delay(np.zeros(200), 0), 50)
        self.assertTrue(np.isnan(metrics.recovery_delay(np.ones(200), 0)))
        self.assertEqual(metrics.recovery_delay(np.zeros(30), 0, window=3, hold=4), 3)
        # the delay is the start of the stable period, not its confirmation
        errors = np.r_[np.ones(10), np.zeros(20)]
        self.assertEqual(metrics.recovery_delay(errors, 0, window=2, hold=3), 12)
        # a dip shorter than hold does not count as recovery
        errors = np.r_[np.ones(5), np.zeros(3), np.ones(5), np.zeros(20)]
        self.assertEqual(metrics.recovery_delay(errors, 0, window=1, hold=4), 14)
        errors, row = metrics.seed_metrics(truth, truth, np.zeros(400), .1,
            evaluation_start=80, warmup=30, lam1=2, lam2=.1, calibration_size=80)
        self.assertEqual(row['mean_error_after_calibration'], 0)
        self.assertAlmostEqual(row['mean_captured_fraction'], 2 / 2.1)

    def test_csv_update_and_preserved_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.csv'
            settings = dict(n_samples=20, seeds=[0, 1], lam1=5., lam2=.5,
                theta0=0., theta1=np.pi / 3, switch_step=10, warmup=3,
                smooth_window=2, recovery_threshold=.05, recovery_hold=2, results_csv=path)
            runs = np.zeros((2, 20))
            angles = np.zeros_like(runs)
            benchmark.save_results('Test PCA', 'stationary', runs, [.1, .2], settings=settings, angle_runs=angles)
            benchmark.save_results('Test PCA', 'stationary', runs, [.2, .3], settings=settings, angle_runs=angles)
            with path.open(newline='', encoding='utf-8') as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(float(rows[0]['mean_runtime_seconds']), .25)
            benchmark.save_results('Test PCA', 'abrupt', runs, [.1, .2], settings=settings, angle_runs=angles)
            with path.open(newline='', encoding='utf-8') as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(float(rows[1]['median_recovery_delay']), 2)



class MathematicalChecks(unittest.TestCase):
    """Independent small reference calculations, separate from migration snapshots."""

    def setUp(self):
        self.X = np.random.default_rng(17).normal(size=(40, 2))
        self.gains = [.005, .02, .1]

    def assert_same_axis(self, actual, expected):
        # Compare axes without requiring the same eigenvector sign.
        np.testing.assert_allclose(abs(actual @ expected), 1, atol=1e-10)

    def test_cumulative_at_every_observation(self):
        directions, final = methods.cumulative_pca(self.X)
        for index, direction in enumerate(directions):
            prefix = self.X[:index + 1]
            C = prefix.T @ prefix / len(prefix)
            self.assert_same_axis(direction, np.linalg.eigh(C)[1][:, -1])
        np.testing.assert_allclose(final, C, atol=1e-12)

    def test_forgetting_at_every_observation(self):
        for alpha in self.gains:
            directions, final = methods.forgetting_pca(self.X, alpha)
            for index, direction in enumerate(directions):
                # Explicit weighted sum, independent of the recursive implementation.
                C = sum(alpha * (1 - alpha)**(index - j) * np.outer(x, x)
                        for j, x in enumerate(self.X[:index + 1]))
                self.assert_same_axis(direction, np.linalg.eigh(C)[1][:, -1])
            np.testing.assert_allclose(final, C, atol=1e-12)

    def test_sliding_at_every_observation(self):
        for window in [1, 7, len(self.X) + 5]:
            directions, final = methods.sliding_window_pca(self.X, window)
            for index, direction in enumerate(directions):
                block = self.X[max(0, index + 1 - window):index + 1]
                C = block.T @ block / len(block)
                self.assert_same_axis(direction, np.linalg.eigh(C)[1][:, -1])
            np.testing.assert_allclose(final, C, atol=1e-12)

    def test_window_cov_update_downdate_equals_recompute(self):
        # The update/downdate recursion must give the same estimate as recomputing the window.
        for window in [1, 7, len(self.X) + 5]:
            for recompute_every in [None, 5]:
                with self.subTest(window=window, recompute_every=recompute_every):
                    expected_q, expected_C = methods.sliding_window_pca(self.X, window)
                    q, C = methods.sliding_window_cov_pca(self.X, window, recompute_every)
                    np.testing.assert_allclose(C, expected_C, atol=1e-12)
                    np.testing.assert_allclose(metrics.axis_error(q, expected_q), 0, atol=1e-10)

    def test_window_rank2_svd_equals_recompute(self):
        # In 2D a rank-2 SVD update/downdate is exact.
        for window in [1, 7, len(self.X) + 5]:
            with self.subTest(window=window):
                expected_q, expected_C = methods.sliding_window_pca(self.X, window)
                q, C = methods.sliding_window_svd_pca(self.X, window, rank=2)
                np.testing.assert_allclose(C, expected_C, atol=1e-10)
                np.testing.assert_allclose(metrics.axis_error(q, expected_q), 0, atol=1e-8)

    def test_window_rank1_svd_at_every_observation(self):
        # Reference: explicit rank-one truncation after the update and after the downdate.
        streams = [self.X, np.zeros((5, 2)), np.tile([2., 1.], (8, 1)),
                   np.array([[3., 0.], [0., 2.], [1., 4.], [2., -1.]])]
        for window in [1, 3, 10]:
            for stream_index, X in enumerate(streams):
                with self.subTest(window=window, stream=stream_index):
                    directions, final = methods.sliding_window_svd_pca(X, window)
                    A = np.zeros((2, 2))
                    for index, x in enumerate(X):
                        changes = [np.outer(x, x)]
                        if index >= window:
                            changes.append(-np.outer(X[index - window], X[index - window]))
                        for change in changes:
                            values, vectors = np.linalg.eigh(A + change)
                            A = max(values[-1], 0.) * np.outer(vectors[:, -1], vectors[:, -1])
                            gap = values[-1] - values[0]
                        if gap > 1e-9 and np.trace(A) > 1e-12:
                            self.assert_same_axis(directions[index], vectors[:, -1])
                    np.testing.assert_allclose(final, A / min(len(X), window), atol=1e-10)
                    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-12)

    def test_window_rank1_without_downdates_equals_incremental(self):
        # A window longer than the stream never downdates: it is rank-1 incremental PCA.
        q, C = methods.sliding_window_svd_pca(self.X, len(self.X) + 5)
        expected_q, expected_C = methods.incremental_pca(self.X)
        np.testing.assert_allclose(C, expected_C, atol=1e-12)
        np.testing.assert_allclose(metrics.axis_error(q, expected_q), 0, atol=1e-10)

    def test_window_cov_with_beta_at_every_observation(self):
        # Truncated exponential window (Badeau et al. 2005, eq. 2): explicit weighted sum, normalized.
        for window in [1, 7, len(self.X) + 5]:
            for beta in [1., .9]:
                for recompute_every in [None, 5]:
                    with self.subTest(window=window, beta=beta, recompute_every=recompute_every):
                        directions, final = methods.sliding_window_cov_pca(self.X, window, recompute_every, beta=beta)
                        for index, direction in enumerate(directions):
                            start = max(0, index + 1 - window)
                            weights = beta ** np.arange(index - start, -1, -1)
                            block = self.X[start:index + 1]
                            C = (block * weights[:, None]).T @ block / weights.sum()
                            self.assert_same_axis(direction, np.linalg.eigh(C)[1][:, -1])
                        np.testing.assert_allclose(final, C, atol=1e-12)

    def test_window_cov_beta_one_is_rectangular_window(self):
        q, C = methods.sliding_window_cov_pca(self.X, 7, beta=1.)
        expected_q, expected_C = methods.sliding_window_pca(self.X, 7)
        np.testing.assert_allclose(C, expected_C, atol=1e-12)
        np.testing.assert_allclose(metrics.axis_error(q, expected_q), 0, atol=1e-10)

    @staticmethod
    def tw_api_table_ii(X, window, beta):
        # Direct transcription of Table II (TW-API) with matrices, r = 1, explicit rescaling of the
        # stored compressed vectors: an independent reference for the scalar implementation.
        W = np.array([[1.], [0.]])
        Z = np.eye(1)
        X_buffer = np.zeros((2, window))
        V_hat = np.zeros((1, window))
        J_inv = np.diag([1., -beta ** (-window)])
        directions = np.zeros((len(X), 2))
        for t, x in enumerate(X):
            x_old = X_buffer[:, 0].copy()
            v_hat_old = V_hat[:, 0].copy()
            y = W.T @ x
            X_buffer = np.column_stack((X_buffer[:, 1:], x))
            Y_hat = np.column_stack((V_hat[:, 1:], y))
            v = W.T @ x_old
            x_pair = np.column_stack((x, x_old))
            y_hat_pair = np.column_stack((y, v_hat_old))
            y_pair = np.column_stack((y, v))
            h = Z @ y_hat_pair
            g = h @ np.linalg.inv(beta * J_inv + y_pair.T @ h)
            e = x_pair - W @ y_pair
            M = np.eye(1) + g @ (e.T @ e) @ g.T
            Theta = M ** -0.5
            Z = (1 / beta) * Theta.T @ (np.eye(1) - g @ y_pair.T) @ Z @ np.linalg.inv(Theta.T)
            W = (W + e @ g.T) @ Theta
            V_hat = Theta.T @ Y_hat
            directions[t] = W[:, 0]
        return directions, 1 / (Z[0, 0] * sum(beta ** k for k in range(min(len(X), window))))

    def test_tw_api_matches_table_ii(self):
        streams = [data.generate_stationary(3, n_samples=300)[0], data.generate_abrupt(3, n_samples=300, switch_step=150)[0]]
        for stream_index, X in enumerate(streams):
            for window in [10, 50]:
                for beta in [1., .95]:
                    with self.subTest(stream=stream_index, window=window, beta=beta):
                        q, eigenvalue = methods.tw_api_pca(X, window, beta)
                        expected_q, expected_eigenvalue = self.tw_api_table_ii(X, window, beta)
                        np.testing.assert_allclose(q, expected_q, atol=1e-10)
                        np.testing.assert_allclose(eigenvalue, expected_eigenvalue, rtol=1e-10)

    def test_tw_api_follows_exact_window(self):
        # TW-API approximates one power iteration per step: in stationary data it stays very close to the
        # exact window direction, and its energy estimate to the exact leading eigenvalue.
        X, _, _ = data.generate_stationary(0)
        q, eigenvalue = methods.tw_api_pca(X, 100)
        exact_q, exact_C = methods.sliding_window_cov_pca(X, 100)
        gap = metrics.axis_error(q, exact_q)[200:]
        self.assertLess(gap.mean(), 1e-4)
        self.assertLess(gap.max(), 1e-3)
        np.testing.assert_allclose(eigenvalue, np.linalg.eigvalsh(exact_C)[-1], rtol=1e-2)
        np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1, atol=1e-12)
        # after an abrupt change it catches up with the exact window once the old samples have left
        X, _, _ = data.generate_abrupt(0)
        q, _ = methods.tw_api_pca(X, 100)
        exact_q, _ = methods.sliding_window_cov_pca(X, 100)
        self.assertLess(metrics.axis_error(q, exact_q)[-500:].max(), 1e-3)

    def test_tw_api_rescaling_matches_table_ii(self):
        # with a tiny threshold the running product is folded into the stored coordinates many times;
        # the result must still be the matrix transcription of Table II
        X = data.generate_abrupt(4, n_samples=300, switch_step=150)[0]
        default = methods.TW_API_RESCALE_LOG
        try:
            methods.TW_API_RESCALE_LOG = 1e-3
            q, eigenvalue = methods.tw_api_pca(X, 10, .95)
        finally:
            methods.TW_API_RESCALE_LOG = default
        expected_q, expected_eigenvalue = self.tw_api_table_ii(X, 10, .95)
        np.testing.assert_allclose(q, expected_q, atol=1e-10)
        np.testing.assert_allclose(eigenvalue, expected_eigenvalue, rtol=1e-10)

    def test_tw_api_long_stream_short_window(self):
        # without rescaling, exp(-log_scale) overflowed after about 9e4 steps for w = 5
        X, _, _ = data.generate_stationary(0, n_samples=150_000)
        q, eigenvalue = methods.tw_api_pca(X, 5)
        self.assertTrue(np.isfinite(q).all() and np.isfinite(eigenvalue))
        np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1, atol=1e-12)
        exact_q, _ = methods.sliding_window_cov_pca(X[-2000:], 5)
        self.assertLess(metrics.axis_error(q[-1990:], exact_q[-1990:]).mean(), 0.05)

    def test_tw_api_reports_breakdown_for_very_short_windows(self):
        # with w = 1 the window matrix changes completely at every step and the approximation fails
        X, _, _ = data.generate_abrupt(0)
        with self.assertRaises(FloatingPointError):
            methods.tw_api_pca(X, 1)

    @staticmethod
    def oja_reference(X, gains):
        # Eq. (2)-(3) of the proposal written literally with matrices, and eq. (4) for the innovation.
        q = np.array([1., 0.])
        directions, innovations = [], []
        for x, alpha in zip(X, gains):
            P = np.eye(2) - np.outer(q, q)
            innovations.append(np.linalg.norm(P @ x) ** 2)
            q_tilde = q + alpha * P @ np.outer(x, x) @ q
            q = q_tilde / np.linalg.norm(q_tilde)
            directions.append(q)
        return np.array(directions), np.array(innovations)

    def test_oja_matches_proposal_equations(self):
        streams = [self.X, data.generate_abrupt(2, n_samples=120, switch_step=60)[0]]
        for stream_index, X in enumerate(streams):
            gain_cases = [np.full(len(X), a) for a in [.001, .02, .3]]
            gain_cases.append(np.linspace(.2, .001, len(X)))
            for case_index, gains in enumerate(gain_cases):
                with self.subTest(stream=stream_index, gains=case_index):
                    q, r = methods.oja_pca(X, gains)
                    expected_q, expected_r = self.oja_reference(X, gains)
                    np.testing.assert_allclose(q, expected_q, atol=1e-12)
                    np.testing.assert_allclose(r, expected_r, atol=1e-12)

    def test_oja_scalar_gain_equals_constant_sequence(self):
        q, r = methods.oja_pca(self.X, .05)
        q_seq, r_seq = methods.oja_pca(self.X, np.full(len(self.X), .05))
        np.testing.assert_array_equal(q, q_seq)
        np.testing.assert_array_equal(r, r_seq)

    def test_oja_innovation_uses_previous_direction(self):
        # r_n = ||x_n||^2 - (q_{n-1} . x_n)^2, with the direction available before x_n is used
        q, r = methods.oja_pca(self.X, .05)
        previous = np.vstack(([1., 0.], q[:-1]))
        expected = np.sum(self.X ** 2, axis=1) - np.sum(previous * self.X, axis=1) ** 2
        np.testing.assert_allclose(r, expected, atol=1e-12)
        self.assertTrue((r >= 0).all())

    def test_oja_stationary_noise_matches_linearized_model(self):
        # Linearized Oja: angle error relaxes with kappa = lam1 - lam2 and noise sigma^2 = lam1 lam2,
        # so the stationary axis error is about alpha lam1 lam2 / (2 (lam1 - lam2)) for a small gain.
        lam1, lam2, alpha = 5., .5, .02 / 4.5
        errors = []
        for seed in range(20):
            X, _, truth = data.generate_stationary(seed, n_samples=4000, lam1=lam1, lam2=lam2)
            q, _ = methods.oja_pca(X, alpha)
            errors.append(metrics.axis_error(q, truth)[1000:].mean())
        predicted = alpha * lam1 * lam2 / (2 * (lam1 - lam2))
        np.testing.assert_allclose(np.mean(errors), predicted, rtol=.2)

    def test_oja_rejects_invalid_arguments(self):
        for gain in [0., -.1, np.nan, np.inf, np.full(3, .1), np.r_[np.full(len(self.X) - 1, .1), 0.]]:
            with self.subTest(gain=gain):
                with self.assertRaises(ValueError):
                    methods.oja_pca(self.X, gain)
        for X in [np.zeros((0, 2)), np.zeros((5, 3)), np.array([[1., np.nan]])]:
            with self.subTest(shape=X.shape):
                with self.assertRaises(ValueError):
                    methods.oja_pca(X, .1)

    def test_window_methods_reject_invalid_arguments(self):
        for fn in [methods.sliding_window_cov_pca, methods.sliding_window_svd_pca, methods.tw_api_pca]:
            for window in [0, -3, 2.5]:
                with self.subTest(method=fn.__name__, window=window):
                    with self.assertRaises(ValueError):
                        fn(self.X, window)
        for recompute_every in [0, 1.5]:
            with self.assertRaises(ValueError):
                methods.sliding_window_cov_pca(self.X, 5, recompute_every)
        with self.assertRaises(ValueError):
            methods.sliding_window_svd_pca(self.X, 5, rank=3)
        for beta in [0., -.5, 1.5]:
            with self.subTest(beta=beta):
                with self.assertRaises(ValueError):
                    methods.sliding_window_cov_pca(self.X, 5, beta=beta)
                with self.assertRaises(ValueError):
                    methods.tw_api_pca(self.X, 5, beta)

    def test_rank1_weighted_update_at_every_observation(self):
        # Include zero and collinear observations as well as ordinary random data.
        streams = [self.X, np.zeros((5, 2)), np.tile([2., 1.], (8, 1)),
                   np.array([[3., 0.], [0., 2.], [1., 4.]])]
        for alpha in self.gains + [1.]:
            for X in streams:
                directions, final = methods.rank1_svd_pca(X, alpha)
                A = np.zeros((2, 2))
                for index, x in enumerate(X):
                    B = (1 - alpha) * A + alpha * np.outer(x, x)
                    values, vectors = np.linalg.eigh(B)
                    if values[-1] - values[0] > 1e-12:
                        self.assert_same_axis(directions[index], vectors[:, -1])
                    A = values[-1] * np.outer(vectors[:, -1], vectors[:, -1])
                np.testing.assert_allclose(final, A, atol=1e-10)
                np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-12)


    def test_rank1_three_properties(self):
        X_random = np.random.default_rng(23).normal(size=(25, 2))
        streams = [X_random, np.zeros((5, 2)), np.tile([2.0, 1.0], (8, 1))]

        for alpha in [0.005, 0.02, 0.1, 1.0]:
            for stream_index, X in enumerate(streams):
                reference = np.zeros((2, 2))
                for index, x in enumerate(X):
                    with self.subTest(alpha=alpha, stream=stream_index, observation=index + 1):
                        directions, C = methods.rank1_svd_pca(X[:index + 1], alpha)

                        # 1. all estimated directions have unit length.
                        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-12)
                        # 2. C is symmetric, positive semidefinite and has rank at most one.
                        np.testing.assert_allclose(C, C.T, atol=1e-12)
                        tolerance = 1e-10 * max(1.0, np.linalg.norm(C, ord=2))
                        self.assertGreaterEqual(np.linalg.eigvalsh(C).min(),-tolerance)
                        self.assertLessEqual(np.linalg.matrix_rank(C, tol=tolerance), 1)
                        # 3. compare with the direct weighted update followed by its best rank-one truncation.
                        updated = ((1 - alpha) * reference + alpha * np.outer(x, x))

                        values, vectors = np.linalg.eigh(updated)
                        leading = vectors[:, -1]
                        reference = (values[-1] * np.outer(leading, leading))

                        np.testing.assert_allclose(C, reference, rtol=1e-10, atol=1e-12)

                        # the returned direction represents C.
                        q = directions[-1]
                        np.testing.assert_allclose(C, np.trace(C) * np.outer(q, q), rtol=1e-10, atol=1e-12)

    def test_methods_are_causal_and_do_not_change_input(self):
        X = self.X.copy()
        X.setflags(write=False)
        cases = [(methods.cumulative_pca, ()), (methods.sliding_window_pca, (7,)),
                 (methods.incremental_pca, ()), (methods.sliding_window_cov_pca, (7,)),
                 (methods.sliding_window_svd_pca, (7,)), (methods.sliding_window_cov_pca, (7, None, .9)),
                 (methods.tw_api_pca, (7,)), (methods.tw_api_pca, (7, .9)),
                 (methods.oja_pca, (.02,)), (methods.oja_pca, (.3,))]
        cases += [(fn, (a,)) for fn in [methods.forgetting_pca, methods.rank1_svd_pca]
                  for a in self.gains]
        for fn, args in cases:
            with self.subTest(method=fn.__name__, args=args):
                complete = fn(X, *args)[0]
                prefix = fn(X[:20], *args)[0]
                np.testing.assert_allclose(prefix, complete[:20], atol=1e-12, equal_nan=True)
                valid = np.isfinite(complete).all(axis=1)
                np.testing.assert_allclose(np.linalg.norm(complete[valid], axis=1), 1, atol=1e-12)
        np.testing.assert_array_equal(X, self.X)

    def test_rank2_svd_with_fixed_gain_equals_forgetting(self):
        # In 2D a rank-2 SVD update is exact: it must reproduce forgetting_pca.
        for alpha in self.gains + [1.]:
            with self.subTest(alpha=alpha):
                directions, C = methods.svd_pca(self.X, alpha, rank=2)
                expected_q, expected_C = methods.forgetting_pca(self.X, alpha) if alpha < 1 else (
                    None, np.outer(self.X[-1], self.X[-1]))
                np.testing.assert_allclose(C, expected_C, atol=1e-12)
                if expected_q is not None:
                    np.testing.assert_allclose(metrics.axis_error(directions, expected_q), 0, atol=1e-10)

    def test_rank2_incremental_equals_cumulative(self):
        # alpha_n = 1/n with full rank gives the exact cumulative covariance.
        directions, C = methods.incremental_pca(self.X, rank=2)
        expected_q, expected_C = methods.cumulative_pca(self.X)
        np.testing.assert_allclose(C, expected_C, atol=1e-12)
        np.testing.assert_allclose(metrics.axis_error(directions, expected_q), 0, atol=1e-10)

    def test_rank1_incremental_at_every_observation(self):
        # Reference: cumulative weighted update followed by the best rank-one truncation.
        streams = [self.X, np.zeros((5, 2)), np.tile([2., 1.], (8, 1)),
                   np.array([[3., 0.], [0., 2.], [1., 4.]])]
        for stream_index, X in enumerate(streams):
            with self.subTest(stream=stream_index):
                directions, final = methods.incremental_pca(X)
                A = np.zeros((2, 2))
                for index, x in enumerate(X):
                    alpha = 1 / (index + 1)
                    B = (1 - alpha) * A + alpha * np.outer(x, x)
                    values, vectors = np.linalg.eigh(B)
                    if values[-1] - values[0] > 1e-12:
                        self.assert_same_axis(directions[index], vectors[:, -1])
                    A = values[-1] * np.outer(vectors[:, -1], vectors[:, -1])
                np.testing.assert_allclose(final, A, atol=1e-10)
                np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-12)
                self.assertTrue(np.isfinite(directions).all())

    def test_svd_pca_accepts_gain_sequence(self):
        # A constant gain sequence gives the same result as the scalar gain.
        for alpha in self.gains:
            scalar = methods.svd_pca(self.X, alpha)
            sequence = methods.svd_pca(self.X, np.full(len(self.X), alpha))
            np.testing.assert_allclose(scalar[0], sequence[0], atol=1e-14)
            np.testing.assert_allclose(scalar[1], sequence[1], atol=1e-14)
        # rank1_svd_pca is the fixed-gain rank-one case.
        np.testing.assert_allclose(methods.rank1_svd_pca(self.X, .02)[1],
                                   methods.svd_pca(self.X, .02, rank=1)[1], atol=1e-14)

    def test_svd_pca_rejects_invalid_gains_and_rank(self):
        bad_gains = [0., -.1, 1.5, np.nan, np.full(len(self.X) - 1, .1),
                     np.r_[np.full(len(self.X) - 1, .1), 0.]]
        for gain in bad_gains:
            with self.subTest(gain=gain):
                with self.assertRaises(ValueError):
                    methods.svd_pca(self.X, gain)
        for rank in [0, 3]:
            with self.assertRaises(ValueError):
                methods.svd_pca(self.X, .1, rank=rank)

    def test_new_generators_and_reproducibility(self):
        options = dict(n_samples=60, theta0=.3, lam1=4., lam2=.2, dt=.5)
        for fn, parameters in [(data.generate_smooth_rotation, dict(omega=.04)),
                               (data.generate_drift, dict(Q=.003))]:
            first = fn(9, **options, **parameters)
            second = fn(9, **options, **parameters)
            for a, b in zip(first, second):
                np.testing.assert_array_equal(a, b)
            X, theta, truth = first
            self.assertEqual(X.shape, (60, 2))
            np.testing.assert_allclose(truth, np.column_stack((np.cos(theta), np.sin(theta))))
            # Remove the rotation and compare with the independent Gaussian draws.
            rng = np.random.default_rng(9)
            if fn is data.generate_drift:
                increments = rng.normal(0, np.sqrt(.003 * .5), 59)
                expected_theta = np.r_[.3, .3 + np.cumsum(increments)]
            else:
                expected_theta = .3 + .04 * .5 * np.arange(60)
            np.testing.assert_allclose(theta, expected_theta)
            z = rng.standard_normal((60, 2)) * np.sqrt([4., .2])
            unrotated = np.column_stack((truth[:, 0] * X[:, 0] + truth[:, 1] * X[:, 1],
                                        -truth[:, 1] * X[:, 0] + truth[:, 0] * X[:, 1]))
            np.testing.assert_allclose(unrotated, z, atol=1e-12)
        for fn, parameters in [(data.generate_drift, dict(Q=0)),
                               (data.generate_smooth_rotation, dict(omega=0))]:
            _, theta, _ = fn(3, **options, **parameters)
            np.testing.assert_allclose(theta, .3)

    def test_known_errors_and_population_captured_variance(self):
        theta = np.array([.2, .7, -.4])
        delta = np.deg2rad([0., 30., 90.])
        truth = np.column_stack((np.cos(theta), np.sin(theta)))
        q = np.column_stack((np.cos(theta + delta), np.sin(theta + delta)))
        E = metrics.axis_error(q, truth)
        np.testing.assert_allclose(E, [0., .25, 1.], atol=1e-12)
        np.testing.assert_allclose(metrics.axis_angle_error(q, theta)[:2], [0., 30.], atol=1e-12)
        for direction, angle, error in zip(q, theta, E):
            direct = direction @ data.covariance_at(angle, 5., .5) @ direction
            self.assertAlmostEqual(direct, 5. - 4.5 * error)

    def test_continuous_csv_metrics_and_gain_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'results.csv'
            settings = dict(n_samples=20, seeds=[0, 1], lam1=5., lam2=.5,
                theta0=0., theta1='', switch_step='', warmup=3, smooth_window=2,
                recovery_threshold='', recovery_hold='', results_csv=path)
            runs = np.full((2, 20), .01)
            angles = np.vstack((np.arange(20), np.arange(20) * 2.))
            for case in ['rotation', 'drift']:
                for gain in [.02, .1]:
                    for _ in range(2):
                        benchmark.save_results('Forgetting PCA', case, runs, [.1, .2],
                            alpha=gain, angle_runs=angles, settings=settings)
            with path.open(newline='', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertAlmostEqual(float(row['mean_tracking_angle_error_sd_degrees']),
                                       angles[:, 3:].std(axis=1).mean())
                self.assertAlmostEqual(float(row['mean_tracking_angle_sd_degrees']),
                                       angles[:, 3:].std(axis=1).mean())
                self.assertAlmostEqual(float(row['mean_tracking_angle_mean_degrees']),
                                       angles[:, 3:].mean(axis=1).mean())
                self.assertAlmostEqual(float(row['mean_tracking_angle_rms_degrees']),
                                       np.sqrt((angles[:, 3:] ** 2).mean(axis=1)).mean())
                self.assertEqual(row['median_recovery_delay'], '')
                self.assertEqual(row['mean_stationary_angle_sd_degrees'], '')

    def test_csv_angle_mean_sd_rms(self):
        # a constant offset of 10 degrees is all bias: mean 10, SD 0, RMS 10;
        # an oscillation of +-2 degrees around zero is all noise: mean 0, SD 2, RMS 2
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'results.csv'
            settings = dict(n_samples=20, seeds=[0, 1], lam1=5., lam2=.5,
                theta0=0., theta1=1., switch_step=10, warmup=4, smooth_window=2,
                recovery_threshold=.05, recovery_hold=2, results_csv=path)
            runs = np.full((2, 20), .01)
            lag = np.full((2, 20), 10.)
            noise = np.tile([2., -2.], (2, 10))
            benchmark.save_results('Lag', 'rotation', runs, [.1, .2], angle_runs=lag, settings=settings)
            benchmark.save_results('Noise', 'stationary', runs, [.1, .2], angle_runs=noise, settings=settings)
            # abrupt: no offset before the change, an offset of -11 degrees in the final segment
            abrupt = np.zeros((2, 20))
            abrupt[:, -4:] = -11.
            benchmark.save_results('Offset', 'abrupt', runs, [.1, .2], angle_runs=abrupt, settings=settings)
            with path.open(newline='', encoding='utf-8') as f:
                rows = {row['method']: row for row in csv.DictReader(f)}
        lag_row, noise_row, abrupt_row = rows['Lag'], rows['Noise'], rows['Offset']
        self.assertAlmostEqual(float(lag_row['mean_tracking_angle_mean_degrees']), 10.)
        self.assertAlmostEqual(float(lag_row['mean_tracking_angle_sd_degrees']), 0.)
        self.assertAlmostEqual(float(lag_row['mean_tracking_angle_rms_degrees']), 10.)
        self.assertAlmostEqual(float(noise_row['mean_stationary_angle_mean_degrees']), 0.)
        self.assertAlmostEqual(float(noise_row['mean_stationary_angle_sd_degrees']), 2.)
        self.assertAlmostEqual(float(noise_row['mean_stationary_angle_rms_degrees']), 2.)
        self.assertAlmostEqual(float(abrupt_row['mean_pre_change_angle_rms_degrees']), 0.)
        self.assertAlmostEqual(float(abrupt_row['mean_final_segment_angle_mean_degrees']), -11.)
        self.assertAlmostEqual(float(abrupt_row['mean_final_segment_angle_rms_degrees']), 11.)
        self.assertEqual(abrupt_row['mean_tracking_angle_rms_degrees'], '')

    def test_rank1_csv_records_one_retained_component(self):
        # A failure here identifies incorrect metadata, not incorrect PCA directions.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'results.csv'
            settings = dict(n_samples=10, seeds=[0, 1], lam1=5., lam2=.5,
                theta0=0., theta1=1., switch_step=5, warmup=2, smooth_window=2,
                recovery_threshold=.05, recovery_hold=2, results_csv=path)
            benchmark.save_results('Rank-1 SVD PCA with fixed forgetting', 'stationary',
                np.zeros((2, 10)), [.1, .2], alpha=.02, settings=settings)
            with path.open(newline='', encoding='utf-8') as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row['retained_components'], '1')
            self.assertEqual(row['centering'], 'known zero mean')


if __name__ == '__main__':
    unittest.main()
