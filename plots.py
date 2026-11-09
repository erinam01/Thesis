"""Existing displays and study figures. Runtime remains table-only."""
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from metrics import trailing_mean, recovery_delay, recovery_delays, axis_angle_error

## COMPUTATIONS =======================================================================================
## PLOT DATA ==========================================================================================
## PLOT ERRORS ========================================================================================

def show_table(headers, rows):
    table = '| ' + ' | '.join(headers) + ' |\n'
    table += '| ' + ' | '.join(['---'] * len(headers)) + ' |\n'
    for row in rows:
        values = []
        for value in row:
            values.append(str(value))
        table += '| ' + ' | '.join(values) + ' |\n'
    display(Markdown(table))


def plot_abrupt_data(X, switch_step, theta0, theta1):
    fig, ax = plt.subplots(figsize=(6, 5))

    before = X[:switch_step]
    after = X[switch_step:]

    ax.scatter(before[:, 0], before[:, 1], s=8, alpha=0.25, color='tab:blue', label='Before change')

    line = np.array([-8, 8])
    ax.plot(line * np.cos(theta0), line * np.sin(theta0), 'r--', label='True axis before change')
    ax.plot(line * np.cos(theta1), line * np.sin(theta1), 'g--', label='True axis after change')

    ax.axis('equal')
    ax.set_xlabel('Coordinate 1')
    ax.set_ylabel('Coordinate 2')
    ax.set_title('Abrupt-change observations')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.show()
    return fig, ax


def plot_stationary_data(X, true_angle):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[:, 0], X[:, 1], s=8, alpha=0.25)
    line = np.array([-8, 8])
    ax.plot(line * np.cos(true_angle), line * np.sin(true_angle),
            'r--', label='True axis')
    ax.axis('equal')
    ax.set_xlabel('Coordinate 1')
    ax.set_ylabel('Coordinate 2')
    ax.set_title('Stationary observations')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.show()
    return fig, ax

def plot_smooth_rotation_data(X, theta0, omega, dt):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[:, 0], X[:, 1], s=8, alpha=0.25)
    line = np.array([-8, 8])
    # plot the true axis at the start and end of the rotation
    ax.plot(line * np.cos(theta0), line * np.sin(theta0),
            'r--', label='True axis at start')
    theta_end = theta0 + omega * dt * (len(X) - 1)
    ax.plot(line * np.cos(theta_end), line * np.sin(theta_end),
            'g--', label='True axis at end')
    ax.axis('equal')
    ax.set_xlabel('Coordinate 1')
    ax.set_ylabel('Coordinate 2')
    ax.set_title('Smooth rotation observations')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.show()
    return fig, ax

def plot_drift_data(X, theta):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[:, 0], X[:, 1], s=8, alpha=0.25)
    line = np.array([-8, 8])

    # plot the true axis at the start of the drift
    theta_start = theta[0]
    ax.plot(line * np.cos(theta_start), line * np.sin(theta_start),
            'r--', label='True axis at start')

    # plot the true axis at the end of the drift
    theta_end = theta[-1]
    ax.plot(line * np.cos(theta_end), line * np.sin(theta_end),
            'g--', label='True axis at end')

    ax.axis('equal')
    ax.set_xlabel('Coordinate 1')
    ax.set_ylabel('Coordinate 2')
    ax.set_title('Random drift observations')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.show()
    return fig, ax

def plot_stationary_errors(error_curves, labels, warmup, smooth_window):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    raw_max = 0.0
    smoothed_max = 0.0
    for errors, label in zip(error_curves, labels):
        steps = np.arange(1, len(errors) + 1)
        smoothed = trailing_mean(errors, smooth_window)
        axes[0].plot(steps, errors, label=label)
        axes[1].plot(steps, smoothed, label=label)
        raw_max = max(raw_max, np.nanmax(errors[warmup:]))
        smoothed_max = max(smoothed_max, np.nanmax(smoothed[warmup:]))

    axes[0].set_title('Raw error')
    axes[1].set_title('Trailing average error')
    axes[0].set_ylim(0, max(1e-10, 1.1 * raw_max))
    axes[1].set_ylim(0, max(1e-10, 1.1 * smoothed_max))
    for ax in axes:
        ax.set_xlabel('Observations processed')
        ax.set_ylabel('Axis error E')
        ax.set_xlim(warmup + 1, len(errors))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=9)
    plt.show()
    return fig, axes


def plot_abrupt_errors(error_curves, labels, switch_step, smooth_window,
                       recovery_threshold, recovery_hold):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for errors, label in zip(error_curves, labels):
        steps = np.arange(1, len(errors) + 1)
        axes[0].plot(steps, errors, label=label)
        before = trailing_mean(errors[:switch_step], smooth_window)
        after = trailing_mean(errors[switch_step:], smooth_window)
        smoothed = np.concatenate((before, after))
        line, = axes[1].plot(steps, smoothed, label=label)
        delay = recovery_delay(errors, switch_step, threshold=recovery_threshold,
                               window=smooth_window, hold=recovery_hold)
        if np.isfinite(delay):
            axes[1].axvline(switch_step + delay, linestyle=':',
                            color=line.get_color(), label=label + ': recovery onset')

    axes[0].set_title('Raw error')
    axes[1].set_title('Trailing average error')
    axes[1].axhline(recovery_threshold, color='gray', linestyle=':')
    for ax in axes:
        ax.set_xlabel('Observations processed')
        ax.set_ylabel('Axis error E')
        ax.axvline(switch_step + 1, color='gray', linestyle='--')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=9)
    plt.show()
    return fig, axes


def plot_cumulative_error(total_curves, labels, title, switch_step=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    for total_error, label in zip(total_curves, labels):
        steps = np.arange(1, len(total_error) + 1)  # entry i is the total after observation i + 1
        ax.plot(steps, total_error, label=label)
    if switch_step is not None:
        ax.axvline(switch_step + 1, color='gray', linestyle='--', label='Change')
    ax.set_xlabel('Observations processed')
    ax.set_ylabel('Accumulated axis error')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=9)
    plt.show()
    return fig, ax


def plot_angle_error(angle_curves, labels, title, warmup, switch_step=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    for angles, label in zip(angle_curves, labels):
        steps = np.arange(1, len(angles) + 1)
        ax.plot(steps[warmup:], angles[warmup:], label=label)
    ax.axhline(0, color='black', linestyle='--')
    if switch_step is not None:
        ax.axvline(switch_step + 1, color='gray', linestyle='--', label='Change')
    ax.set_xlim(warmup + 1, len(angles))
    ax.set_xlabel('Observations processed')
    ax.set_ylabel('Signed axis-angle error (degrees)')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=9)
    plt.show()
    return fig, ax


def plot_repeated_errors(run_arrays, labels, title, start=0, warmup=None,
                         switch_step=None, upper_limit=1):
    fig, ax = plt.subplots(figsize=(10, 4))
    for runs, label in zip(run_arrays, labels):
        steps = np.arange(start + 1, runs.shape[1] + 1)
        mean_error = runs[:, start:].mean(axis=0)
        low, high = np.percentile(runs[:, start:], [10, 90], axis=0)
        ax.plot(steps, mean_error, label=label)
        ax.fill_between(steps, low, high, alpha=0.15)
    if warmup is not None:
        ax.set_xlim(warmup + 1, runs.shape[1])
    if switch_step is not None:
        ax.axvline(switch_step + 1, color='gray', linestyle='--')
    ax.set_ylim(0, upper_limit)
    ax.set_xlabel('Observations processed')
    ax.set_ylabel('Axis error E')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.show()
    return fig, ax


def plot_captured_variance(percent_curves, labels, title, best_percent,
                           switch_step=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    for percentages, label in zip(percent_curves, labels):
        steps = np.arange(1, len(percentages) + 1)
        ax.plot(steps, percentages, label=label)
    ax.axhline(best_percent, color='black', linestyle='--', label='True leading axis')
    if switch_step is not None:
        ax.axvline(switch_step + 1, color='gray', linestyle='--', label='Change')
    ax.set_xlabel('Observations processed')
    ax.set_ylabel('Total variance captured (%)')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=9)
    plt.show()
    return fig, ax


## PART 1 BLOCKS (02_sliding_window, 03_oja) ==========================================================
# The same tables and plots as the cells of 01_baselines, one function per block.
#   single:   {value: (directions, raw errors)} on the displayed seed (benchmark.single_run)
#   results:  {value: (runs, angles, runtimes)} on all seeds (benchmark.repeat_runs)
#   settings: the settings dictionary of the regime (warmup, smooth_window, lam1, lam2 and, for the
#             abrupt regime, switch_step, recovery_threshold, recovery_hold)
#   label:    function value -> text shown in tables and legends

def show_single_run(single, scenario, settings, label):
    # SINGLE-RUN TABLE: mean error and temporal SD after warm-up; abrupt: pre/post-change, final segment, delay
    warmup = settings['warmup']
    rows = []
    for value, (directions, errors) in single.items():
        if scenario == 'abrupt':
            switch_step = settings['switch_step']
            delay = recovery_delay(errors, switch_step, threshold=settings['recovery_threshold'],
                                   window=settings['smooth_window'], hold=settings['recovery_hold'])
            rows.append([label(value), f'{errors[warmup:switch_step].mean():.5f}', f'{errors[switch_step:].mean():.5f}',
                         f'{errors[-warmup:].mean():.5f}', delay])
        else:
            rows.append([label(value), f'{errors[warmup:].mean():.5f}', f'{errors[warmup:].std():.5f}'])
    if scenario == 'abrupt':
        show_table(['Method', 'Pre-change mean E', 'Post-change mean E', 'Final-segment mean E', 'Recovery delay'], rows)
    else:
        show_table(['Method', 'Mean error after warm-up', 'Temporal error SD'], rows)


def show_cost(results, label):
    # RUNTIME TABLE OVER THE SEEDS, in seconds
    rows = []
    for value, (_, _, runtimes) in results.items():
        rows.append([label(value), runtimes.mean(), runtimes.min(), runtimes.max(), runtimes.std(ddof=1)])
    show_table(['Method', 'Mean seconds', 'Shortest seconds', 'Longest seconds', 'SD seconds'], rows)


def show_plots(single, results, theta, scenario, settings, title, label):
    # THE FIVE PLOTS OF A BLOCK: raw and trailing error, cumulative error, signed angle and captured
    # variance on the displayed seed (true angles theta), then the across-seed mean error
    warmup, smooth_window = settings['warmup'], settings['smooth_window']
    lam1, lam2 = settings['lam1'], settings['lam2']
    labels = [label(value) for value in single]
    errors = [e for _, e in single.values()]
    directions = [d for d, _ in single.values()]
    abrupt = scenario == 'abrupt'
    switch_step = settings['switch_step'] if abrupt else None

    # Raw axis error and its trailing average: one seed, two panels.
    if abrupt:
        plot_abrupt_errors(errors, labels, switch_step, smooth_window, settings['recovery_threshold'],
                           settings['recovery_hold'])
    else:
        plot_stationary_errors(errors, labels, warmup, smooth_window)

    # Accumulated raw error from observation 1, including the initial learning period.
    total_curves = [np.cumsum(e) for e in errors]
    for total_error, text in zip(total_curves, labels):
        print(text, 'cumulative error from observation 1:', total_error[-1])
    plot_cumulative_error(total_curves, labels, f'Cumulative: {title}, {scenario}', switch_step=switch_step)

    # Signed angular error in degrees; SD excludes warm-up (and, for abrupt, the jump).
    angle_curves = [axis_angle_error(d, theta) for d in directions]
    for angles, text in zip(angle_curves, labels):
        if abrupt:
            print(text, 'stationary angle SD (degrees):', angles[warmup:switch_step].std())
        elif scenario == 'stationary':
            print(text, 'stationary angle SD (degrees):', angles[warmup:].std())
        else:
            print(text, 'tracking angle-error SD (degrees):', angles[warmup:].std())
    plot_angle_error(angle_curves, labels, f'Angle: {title}, {scenario}', warmup, switch_step=switch_step)

    # Ground-truth population variance captured by the estimated axis, as a percentage.
    percent_curves = [100 * (lam1 - (lam1 - lam2) * e) / (lam1 + lam2) for e in errors]
    for percentages, text in zip(percent_curves, labels):
        print(text, 'mean captured variance after warm-up (%):', np.nanmean(percentages[warmup:]))
    best_percent = 100 * lam1 / (lam1 + lam2)
    plot_captured_variance(percent_curves, labels, f'Captured: {title}, {scenario}', best_percent,
                           switch_step=switch_step)

    # Across-seed mean raw error and 10th-90th percentile band, not a confidence interval.
    runs = [results[value][0] for value in single]
    if abrupt:
        plot_repeated_errors(runs, labels, f'{title}: {scenario}, 20 runs', start=0,
                             switch_step=switch_step, upper_limit=1)
    else:
        plot_repeated_errors(runs, labels, f'{title}: {scenario}, 20 runs', start=0, warmup=warmup,
                             upper_limit=0.025 if scenario == 'stationary' else 1)


def show_repeated_measurements(results, scenario, settings, label):
    # REPEATED-RUN TABLE: mean error, SD of run means, mean temporal SD;
    # abrupt: pre/post-change and final error (mean +/- SD), recovered runs, median delay among them
    warmup = settings['warmup']
    rows = []
    for value, (runs, _, _) in results.items():
        if scenario == 'abrupt':
            switch_step = settings['switch_step']
            pre_means = runs[:, warmup:switch_step].mean(axis=1)
            post_means = runs[:, switch_step:].mean(axis=1)
            final_means = runs[:, -warmup:].mean(axis=1)
            delays = recovery_delays(runs, settings)
            recovered = np.isfinite(delays)
            median_delay = float(np.median(delays[recovered])) if recovered.any() else 'Not confirmed'
            rows.append([label(value),
                         f'{pre_means.mean():.5f} +/- {pre_means.std(ddof=1):.5f}',
                         f'{post_means.mean():.5f} +/- {post_means.std(ddof=1):.5f}',
                         f'{final_means.mean():.5f} +/- {final_means.std(ddof=1):.5f}',
                         f'{recovered.sum()}/{len(delays)}', median_delay])
        else:
            run_means = runs[:, warmup:].mean(axis=1)
            rows.append([label(value), f'{run_means.mean():.5f}', f'{run_means.std(ddof=1):.5f}',
                         f'{runs[:, warmup:].std(axis=1).mean():.5f}'])
    if scenario == 'abrupt':
        show_table(['Method', 'Pre-change E: mean +/- SD', 'Post-change E: mean +/- SD',
                    'Final E: mean +/- SD', 'Recovered runs', 'Median delay'], rows)
    else:
        show_table(['Method', 'Mean E', 'SD of run means', 'Mean temporal SD'], rows)


def plot_gain_history(gain_curves, labels, calibration_size, switch_step=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    for gains, label in zip(gain_curves, labels):
        steps = np.arange(1, len(gains) + 1)
        ax.plot(steps, gains, label=label)
    ax.axvline(calibration_size + 0.5, color='black', linestyle=':',
               label='End of calibration')
    if switch_step is not None:
        ax.axvline(switch_step + 1, color='gray', linestyle='--', label='Change')
    ax.set_xlabel('Observations processed')
    ax.set_ylabel('Gain alpha')
    ax.set_title('Adaptive gain')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=9)
    plt.show()
    return fig, ax


def show_rows(rows, columns):
    text = '| ' + ' | '.join(columns) + ' |\n'
    text += '| ' + ' | '.join(['---'] * len(columns)) + ' |\n'
    for row in rows:
        values = [format(row[c], '.6g') if isinstance(row[c], (float, np.floating)) else str(row[c]) for c in columns]
        text += '| ' + ' | '.join(values) + ' |\n'
    display(Markdown(text))


def plot_runs(run_arrays, labels, title, change_index=None, *, evaluation_start=200):
    fig, ax = plt.subplots(figsize=(10, 4))
    for runs, label in zip(run_arrays, labels):
        values = runs[:, evaluation_start:]
        steps = np.arange(evaluation_start + 1, runs.shape[1] + 1)
        low, high = np.percentile(values, [10, 90], axis=0)
        ax.plot(steps, values.mean(axis=0), label=label)
        ax.fill_between(steps, low, high, alpha=0.15)
    if change_index is not None:
        ax.axvline(change_index + 1, color='gray', linestyle='--')
    ax.set_xlabel('Observations processed')
    ax.set_ylabel('Axis error E')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()


def plot_k_sensitivity(tuning_summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    rows = [r for r in tuning_summary if r['method'] == 'adaptive']
    for ax, metric, spread, title in [
        (axes[0], 'stationary_error', 'stationary_sd', 'Stationary error'),
        (axes[1], 'post_change_error', 'post_change_sd', 'Post-change error')
    ]:
        ax.errorbar([r['parameter'] for r in rows], [r[metric] for r in rows],
                    yerr=[r[spread] for r in rows], marker='o', capsize=3)
        ax.set_xscale('log')
        ax.set_xlabel('k')
        ax.set_ylabel('Mean axis error E')
        ax.set_title(title + ': mean +/- across-seed SD')
        ax.grid(True, alpha=0.3)
    plt.show()


def plot_tuning_tradeoff(tuning_summary):
    fig, ax = plt.subplots(figsize=(8, 5))
    offsets = {('adaptive', 0.3961): (-15, 18), ('adaptive', 2.0): (-20, -18),
               ('fixed', 0.02): (8, -15), ('fixed', 0.1): (-110, 8)}
    for row in tuning_summary:
        adaptive = row['method'] == 'adaptive'
        ax.scatter(row['stationary_error'], row['post_change_error'],
                   marker='o' if adaptive else 'x', color='tab:blue' if adaptive else 'tab:orange')
        name = 'k' if adaptive else 'fixed alpha'
        ax.annotate(f"{name}={row['parameter']:g}", (row['stationary_error'], row['post_change_error']),
                    xytext=offsets.get((row['method'], row['parameter']), (5, 5)),
                    textcoords='offset points', fontsize=8)
    ax.margins(x=0.12, y=0.15)
    ax.set_xlabel('Mean stationary error (lower is better)')
    ax.set_ylabel('Mean post-change error (lower is better)')
    ax.set_title('Tuning trade-off: blue adaptive, orange fixed')
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_stationary_gain_comparison(results, k_values, tuning_seeds, calibration_size, alpha_min, alpha_max, n_samples):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for k in k_values:
        gains = results[k]['gains']
        steps = np.arange(1, n_samples + 1)
        axes[0].plot(steps, gains[0], label=f'k={k:g}')
        axes[1].plot(steps, gains.mean(axis=0), label=f'k={k:g}')
    for ax in axes:
        ax.axvline(calibration_size + 0.5, color='black', linestyle=':')
        
        ax.set_xlabel('Observations processed')
        ax.set_ylabel('Gain alpha')
        ax.set_ylim(alpha_min * 0.8, alpha_max * 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title('Single tuning seed: ' + str(tuning_seeds[0]))
    axes[1].set_title('Mean gain across tuning seeds')
    plt.show()


def plot_abrupt_gain_comparison(results, k_values, tuning_seeds, calibration_size, alpha_min, alpha_max, n_samples, switch_step):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for k in k_values:
        gains = results[k]['gains']
        steps = np.arange(1, n_samples + 1)
        axes[0].plot(steps, gains[0], label=f'k={k:g}')
        axes[1].plot(steps, gains.mean(axis=0), label=f'k={k:g}')
    for ax in axes:
        ax.axvline(calibration_size + 0.5, color='black', linestyle=':')
        ax.axvline(switch_step + 1, color='gray', linestyle='--', label='Change')
        ax.set_xlabel('Observations processed')
        ax.set_ylabel('Gain alpha')
        ax.set_ylim(alpha_min * 0.8, alpha_max * 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title('Single tuning seed: ' + str(tuning_seeds[0]))
    axes[1].set_title('Mean gain across tuning seeds')
    plt.show()
