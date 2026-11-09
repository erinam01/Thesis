
### ERRORS AND EVALUATORS ####

import numpy as np


def axis_error(estimated, truth):
    # SIGN-INVARIANT ERROR (1-cosine) = 1 - (dot product of estimated and truth)^2
    alignment = np.sum(estimated * truth, axis=1)
    return np.clip(1 - alignment**2, 0, 1)


def axis_angle_error(directions, true_angles):
    # SIGNED ANGLE ERROR IN DEGREES
    estimated_angles = np.arctan2(directions[:, 1], directions[:, 0])
    difference = estimated_angles - true_angles
    # opposite vectors describe the same axiso; wrap the difference to [-pi/2, pi/2] to reflect that.
    difference = (difference + np.pi / 2) % np.pi - np.pi / 2
    return np.degrees(difference)


def trailing_mean(values, window):
    # SMOOTHED OUTPUT: AVERAGE OF THE PREVIOUS WINDOW OBSERVATIONS, INCLUDING THE CURRENT ONE
    # NaN where a complete trailing window is not yet possible --< use all of the observations
    result = np.full(len(values), np.nan)
    if len(values) >= window:
        result[window - 1:] = np.convolve(values, np.ones(window) / window, mode='valid')
    return result


def recovery_delay(errors, switch_step, threshold=0.05,
                   window=50, hold=50):
    # POST-CHANGE OBSERVATION AT WHICH THE SMOOTHED ERROR BECOMES STABLY LOW; NEEDED FOR ABRUPT CHANGE REGIMES
    # smoothed error = trailing mean of the latest `window` raw errors, restarted at the change
    # recovery = the smoothed error stays at or below `threshold` for `hold` consecutive observations
    # the returned delay is the START of that stable period (counted from the change, first post-change
    # observation = 1), not the observation at which it is confirmed: confirmation comes hold - 1 steps later
    # the smallest possible delay is `window` (the first complete trailing mean); NaN if never recovered
    smooth = trailing_mean(errors[switch_step:], window)
    consecutive = 0
    for index in range(len(smooth)):
        value = smooth[index]
        if np.isfinite(value) and value <= threshold:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= hold:
            # index is the confirming observation; the stable period started hold - 1 observations earlier
            return float(index + 1 - (hold - 1))
    return np.nan


def recovery_delays(runs, settings):
    # RECOVERY DELAY OF EVERY RUN (one row of raw errors per seed), NaN for the runs that never recover
    # settings: the abrupt-regime settings dictionary of the notebook (switch_step, recovery_threshold,
    # smooth_window, recovery_hold)
    return np.array([recovery_delay(errors, settings['switch_step'], threshold=settings['recovery_threshold'],
                                    window=settings['smooth_window'], hold=settings['recovery_hold'])
                     for errors in runs])


def seed_metrics(directions, truth, true_angles, runtime, gains=None,
                 change_index=None, *, evaluation_start=200, warmup=200, lam1=5.0, lam2=0.5, calibration_size=200, recovery_threshold=0.05, smooth_window=50, recovery_hold=50):
    errors = axis_error(directions, truth)
    angles = axis_angle_error(directions, true_angles)
    evaluated = errors[evaluation_start:]
    stationary_stop = len(errors) if change_index is None else change_index
    row = {
        'mean_error_after_calibration': float(evaluated.mean()),
        'cumulative_error_after_calibration': float(evaluated.sum()),
        'final_segment_mean_error': float(errors[-warmup:].mean()),
        'stationary_angle_sd_degrees': float(angles[evaluation_start:stationary_stop].std()),
        'mean_captured_fraction': float((lam1 - (lam1 - lam2) * evaluated).mean() / (lam1 + lam2)),
        'runtime_seconds': float(runtime),
        'post_change_mean_error': '',
        'recovery_delay': '',
        'recovered': '',
        'mean_adaptive_gain': '',
    }
    if change_index is not None:
        row['post_change_mean_error'] = float(errors[change_index:].mean())
        delay = recovery_delay(errors, change_index, threshold=recovery_threshold, window=smooth_window, hold=recovery_hold)
        row['recovered'] = int(np.isfinite(delay))
        row['recovery_delay'] = float(delay) if np.isfinite(delay) else ''
    if gains is not None:
        row['mean_adaptive_gain'] = float(gains[calibration_size:].mean())
    return errors, row


def summarize(rows):
    result = {}
    fields = ['mean_error_after_calibration', 'cumulative_error_after_calibration',
              'final_segment_mean_error', 'stationary_angle_sd_degrees',
              'mean_captured_fraction', 'runtime_seconds', 'post_change_mean_error',
              'mean_adaptive_gain']
    for field in fields:
        values = np.array([r[field] for r in rows if r[field] != ''], dtype=float)
        result[field] = float(values.mean()) if len(values) else ''
        result[field + '_sd_across_seeds'] = float(values.std(ddof=1)) if len(values) > 1 else ''
    times = [r['runtime_seconds'] for r in rows]
    result['min_runtime_seconds'] = min(times)
    result['max_runtime_seconds'] = max(times)
    attempted = [r for r in rows if r['recovered'] != '']
    delays = [r['recovery_delay'] for r in attempted if r['recovery_delay'] != '']
    result['recovered_runs'] = sum(r['recovered'] for r in attempted) if attempted else ''
    result['recovery_rate'] = result['recovered_runs'] / len(attempted) if attempted else ''
    result['median_recovery_delay_among_recovered'] = float(np.median(delays)) if delays else ''
    return result
