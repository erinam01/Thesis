###### SAVE RESULTS INTO CSV FILE ######


import csv
import time
from pathlib import Path
import numpy as np
from metrics import recovery_delay, axis_error, axis_angle_error


## RUNNING THE METHODS (used by the Part 1 blocks of 02_sliding_window and 03_oja) =====================
# streams: one regime, as a dictionary seed -> (X, theta, truth), in the order of the seeds
# fn(X, value): an estimator with one parameter (a gain or a window length), returning (directions, ...)

def run_method(fn, streams):
    # RUN fn(X) ON EVERY STREAM OF ONE REGIME; returns errors, signed angles (degrees) and runtimes,
    # one row per seed. Only the PCA calculation is timed, not generation or evaluation.
    runs, angles, runtimes = [], [], []
    for X_run, theta_run, truth_run in streams.values():
        start = time.perf_counter()
        directions, _ = fn(X_run)
        runtimes.append(time.perf_counter() - start)
        runs.append(axis_error(directions, truth_run))
        angles.append(axis_angle_error(directions, theta_run))
    return np.array(runs), np.array(angles), np.array(runtimes)


def single_run(fn, values, stream):
    # ONE STREAM (the displayed seed), ONE RUN PER PARAMETER VALUE
    # returns {value: (directions, raw axis errors)}
    X, _, truth = stream
    out = {}
    for value in values:
        directions = fn(X, value)[0]
        out[value] = (directions.copy(), axis_error(directions, truth))
    return out


def repeat_runs(fn, values, streams, scenario, settings, method_name, parameter):
    # ALL SEEDS, ONE ROW PER PARAMETER VALUE SAVED IN THE CSV OF settings['results_csv']
    # parameter: name of the CSV column that holds the value ('alpha' or 'window_size')
    # returns {value: (runs, angles, runtimes)}
    out = {}
    for value in values:
        runs, angles, runtimes = run_method(lambda X: fn(X, value), streams)
        out[value] = (runs, angles, runtimes)
        save_results(method_name, scenario, runs, runtimes, angle_runs=angles, settings=settings,
                     **{parameter: value})
    return out


def best_time(fn, repeats=3):
    # SHORTEST OF `repeats` TIMINGS OF fn(), in seconds (the minimum is the least disturbed by other processes)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def save_results(method, data_case, runs, runtimes, alpha='', angle_runs=None, window_size='', *, settings):
    assert isinstance(runs, np.ndarray) and runs.ndim == 2, 'Pass one array of runs, not the gain dictionary'
    assert len(runtimes) == len(runs), 'Need exactly one runtime per seed'
    row = {
        'method': method,
        'scenario': data_case,
        'alpha': alpha,
        'window_size': window_size,
        'n_samples': settings['n_samples'],
        'n_runs': len(runs),
        'seeds': ';'.join(str(s) for s in settings['seeds']),
        'lam1': settings['lam1'],
        'lam2': settings['lam2'],
        'spectral_gap': settings['lam1'] - settings['lam2'],
        'theta0_radians': settings['theta0'],
        'theta1_radians': settings['theta1'],
        'switch_step': settings['switch_step'],
        'warmup': settings['warmup'],
        'final_segment_length': settings['warmup'],
        'smooth_window': settings['smooth_window'],
        'recovery_threshold': settings['recovery_threshold'],
        'recovery_hold': settings['recovery_hold'],
        'centering': 'known zero mean',
        'retained_components': 1 if method in (
            'Incremental PCA (rank-1 SVD)',
            'Rank-1 SVD PCA with fixed forgetting',
            'Sliding-window PCA (rank-1 SVD)',
            'Sliding-window PCA (TW-API)',
            'Oja PCA'
        ) else 2,
        'numpy_version': np.__version__,
    }

    identity_fields = list(row)

    evaluated = runs[:, settings['warmup']:]
    run_means = evaluated.mean(axis=1)

    # ERRORS
    row['mean_error_after_warmup'] = run_means.mean()
    row['sd_of_run_mean_errors'] = run_means.std(ddof=1)
    row['mean_temporal_error_sd'] = evaluated.std(axis=1).mean()
    row['mean_cumulative_error_after_warmup'] = evaluated.sum(axis=1).mean()
    # every method has an estimate from the first observation onward
    row['cumulative_start_observation'] = 1
    row['mean_cumulative_error_common_period'] = runs.sum(axis=1).mean()
    captured = settings['lam1'] - (settings['lam1'] - settings['lam2']) * evaluated
    row['mean_captured_variance_after_warmup'] = captured.mean()
    row['mean_captured_fraction_after_warmup'] = captured.mean() / (settings['lam1'] + settings['lam2'])
    # Signed angular error, summarized over a period of each run and then averaged over the runs:
    # - mean: the systematic offset (lag) of the estimate; signed, so offsets of opposite sign cancel
    # - SD:   the oscillation around the run's own mean; it does not show a systematic offset
    # - RMS:  the total angular error, RMS^2 = mean^2 + SD^2 within each run
    # Periods: stationary and rotation/drift after warm-up; abrupt before the change, and in the
    # final segment (last `warmup` observations), where a persistent offset after the change shows.
    angle_columns = {
        'stationary': 'stationary_angle',
        'abrupt': 'pre_change_angle',
        'abrupt_final': 'final_segment_angle',
        'tracking': 'tracking_angle',
    }
    for prefix in angle_columns.values():
        for stat in ('mean', 'sd', 'rms'):
            row[f'mean_{prefix}_{stat}_degrees'] = ''
    # older column name for the tracking SD, kept so that existing tables still find it
    row['mean_tracking_angle_error_sd_degrees'] = ''

    def angle_summary(prefix, angles):
        row[f'mean_{prefix}_mean_degrees'] = angles.mean(axis=1).mean()
        row[f'mean_{prefix}_sd_degrees'] = angles.std(axis=1).mean()
        row[f'mean_{prefix}_rms_degrees'] = np.sqrt((angles ** 2).mean(axis=1)).mean()

    if angle_runs is not None:
        if data_case == 'stationary':
            angle_summary('stationary_angle', angle_runs[:, settings['warmup']:])
        elif data_case == 'abrupt':
            angle_summary('pre_change_angle', angle_runs[:, settings['warmup']:settings['switch_step']])
            angle_summary('final_segment_angle', angle_runs[:, -settings['warmup']:])
        elif data_case in ('rotation', 'drift'):
            angle_summary('tracking_angle', angle_runs[:, settings['warmup']:])
            row['mean_tracking_angle_error_sd_degrees'] = row['mean_tracking_angle_sd_degrees']
    row['mean_final_sample_error'] = runs[:, -1].mean()
    row['mean_final_segment_error'] = runs[:, -settings['warmup']:].mean()
    row['sd_of_final_segment_mean_errors'] = runs[:, -settings['warmup']:].mean(axis=1).std(ddof=1)

    row['mean_pre_change_error'] = ''
    row['sd_of_pre_change_mean_errors'] = ''
    row['mean_post_change_error'] = ''
    row['sd_of_post_change_mean_errors'] = ''
    row['recovered_runs'] = ''
    row['recovery_rate'] = ''
    row['median_recovery_delay'] = ''
    row['mean_recovery_delay'] = ''
    row['min_recovery_delay'] = ''
    row['max_recovery_delay'] = ''

    # runtimes
    row['mean_runtime_seconds'] = float(np.mean(runtimes))
    row['min_runtime_seconds'] = float(np.min(runtimes))
    row['max_runtime_seconds'] = float(np.max(runtimes))
    row['sd_runtime_seconds'] = float(np.std(runtimes, ddof=1)) # for sample standard deviation

    # Standard deviations use two conventions on purpose:
    # - across runs, ddof=1: the runs are a sample of possible streams, so the sample SD divides by n - 1
    # - over time within one run, ddof=0: it describes that trajectory and does not estimate anything

    if data_case == 'abrupt':
        pre = runs[:, settings['warmup']:settings['switch_step']].mean(axis=1)
        post = runs[:, settings['switch_step']:].mean(axis=1)
        row['mean_pre_change_error'] = pre.mean()
        row['sd_of_pre_change_mean_errors'] = pre.std(ddof=1)
        row['mean_post_change_error'] = post.mean()
        row['sd_of_post_change_mean_errors'] = post.std(ddof=1)
        delays = []
        for errors in runs:
            delay = recovery_delay(errors, settings['switch_step'], threshold=settings['recovery_threshold'], window=settings['smooth_window'], hold=settings['recovery_hold'])
            if np.isfinite(delay):
                delays.append(delay)
        row['recovered_runs'] = len(delays)
        row['recovery_rate'] = len(delays) / len(runs)
        if len(delays) > 0:
            row['median_recovery_delay'] = float(np.median(delays))
            row['mean_recovery_delay'] = float(np.mean(delays))
            row['min_recovery_delay'] = min(delays)
            row['max_recovery_delay'] = max(delays)

    saved_rows = []
    if settings['results_csv'].exists():
        with settings['results_csv'].open(newline='', encoding='utf-8') as file:
            for old_row in csv.DictReader(file):
                matches = True
                for name in identity_fields:
                    if old_row.get(name, '') != str(row[name]):
                        matches = False
                        break
                if not matches:
                    saved_rows.append(old_row)
    saved_rows.append(row)

    with settings['results_csv'].open('w', newline='', encoding='utf-8') as file:
        #preserve older CSV columns when this version adds metrics
        columns = list(row)
        for old_row in saved_rows:
            for column in old_row:
                if column not in columns:
                    columns.append(column)
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(saved_rows)
    path = settings['results_csv'].resolve()
    print(f'Saved {method}, {data_case}, alpha={alpha} to {path}')


def write_rows(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print('Saved:', Path(path).resolve())
