"""Existing PCA estimators. No estimator receives synthetic ground truth."""
import numpy as np


def leading_direction(covariance, previous):
    # FINDS THE LEADING EIGENVECTOR OF A COVARIANCE MATRIX, WITH SIGN CONSISTENCY
    # eigh is designed for symmetric matrices such as covariance matrices.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)  # from smalles to largest
    if np.isclose(eigenvalues[-1], eigenvalues[0], rtol=1e-12, atol=1e-14):
        return previous.copy() # means no unique leading direction, return previous
    q = eigenvectors[:, -1]
    # q and -q describe the same axis; choose the sign closest to the previous q.
    if np.dot(q, previous) < 0: # - q is closer to previous than q
        q = -q
    return q


def cumulative_pca(X):
    # UPDATES ZERO MEAN COVARIANCE MATRIX WITH EACH NEW OBSERVATION, THEN FINDS LEADING DIRECTION
    # ALL PREVIOUS OBSERVATIONS CONTRIBUTE EQUALLY TO THE COVARIANCE ESTIMATE
    covariance = np.zeros((2, 2)) # EMPTy inizialization of covariance matrix
    q = np.array([1.0, 0.0])  # arbitrary initial axis
    directions = np.zeros((len(X), 2))
    for index in range(len(X)):
        x = X[index]
        n = index + 1  # count how many samples have been processed so far (python starts from zero)
        covariance += (np.outer(x, x) - covariance) / n # S = 1/n(x*x.t)
        q = leading_direction(covariance, q)
        directions[index] = q
    return directions, covariance


def forgetting_pca(X, alpha):
    # UPDATES ZERO MEAN COVARIANCE MATRIX WITH EACH NEW OBSERVATION, THEN FINDS LEADING DIRECTION
    # PREVIOUS OBSERVATIONS LOSE INFLUENCE EXPONENTIALLY, HOWEVER THE RATE OF FORGETTING IS FIXED (ALPHA IS CONSTANT)

    if not 0 < alpha < 1:
        raise ValueError('alpha must be in (0, 1)')

    covariance = np.zeros((2, 2))
    q = np.array([1.0, 0.0])
    directions = np.zeros((len(X), 2))
    for index in range(len(X)):
        x = X[index]
        # Retain a fraction of the previous matrix and add the newest contribution.
        covariance = (1 - alpha) * covariance + alpha * np.outer(x, x)
        q = leading_direction(covariance, q)
        directions[index] = q
    return directions, covariance


def incremental_pca(X, rank=1, tol=1e-12):
    # INCREMENTAL (CUMULATIVE) PCA WITH AN SVD UPDATE: ALL OBSERVATIONS HAVE EQUAL WEIGHT
    # it is svd_pca with the gain alpha_n = 1/n, i.e. C_n = (1 - 1/n) C_{n-1} + (1/n) x_n x_n^T
    # with rank=1 the second component is discarded after each update (truncated, like
    # sklearn's IncrementalPCA with n_components=1); rank=2 is exact and is used only in the tests.
    # known zero population mean, as for the other methods
    X = np.asarray(X, dtype=float)
    gains = 1.0 / np.arange(1, len(X) + 1)
    return svd_pca(X, gains, rank=rank, tol=tol)


def sliding_window_pca(X, window_size):
    # UPDATES ZERO MEAN COVARIANCE MATRIX WITH EACH NEW OBSERVATION, THEN FINDS LEADING DIRECTION
    # ONLY THE MOST RECENT WINDOW OF OBSERVATIONS IS USED TO ESTIMATE THE COVARIANCE MATRIX, PREVIOUS ONES ARE IGNORED COMPLETELY
    if window_size < 1:
        raise ValueError('window_size must be at least 1')
    if len(X) == 0:
        raise ValueError('X must contain at least one observation')

    directions = np.zeros((len(X), 2))
    q = np.array([1.0, 0.0])
    for index in range(len(X)):
        start = max(0, index + 1 - window_size)
        window = X[start:index + 1]
        covariance = window.T @ window / len(window)
        q = leading_direction(covariance, q)
        directions[index] = q
    return directions, covariance


def svd_rank_one_modify(U, s2, x, sign, rank, tol=1e-12):
    # ONE RANK-ONE SVD MODIFICATION (Brand, 2006): retained memory U diag(s2) U^T, vector x
    # computes rank_r(U diag(s2) U^T + sign * x x^T) without forming the full matrix
    #   sign = +1 -> update   (an observation enters)
    #   sign = -1 -> downdate (an observation leaves)
    # U: (2, rank) orthonormal columns, s2: (rank,) retained eigenvalues (squared singular values)

    # split x into a part inside the retained subspace and a perpendicular part
    projection = U.T @ x
    residual = x - U @ projection
    residual_norm = np.linalg.norm(residual)

    if U.shape[1] < len(x) and residual_norm > tol:
        # enlarge the basis with the normalized residual
        basis = np.column_stack((U, residual / residual_norm))
        z = np.append(projection, residual_norm)
        s2 = np.append(s2, 0.0)
    else:
        # x lies inside the retained subspace (always true for full rank)
        basis = U
        z = projection

    # small symmetric matrix in the (retained subspace + residual) basis
    small_matrix = np.diag(s2) + sign * np.outer(z, z)
    values, vectors = np.linalg.eigh(small_matrix)  # ascending order

    # keep only the largest `rank` eigenpairs (truncation when rank < basis size)
    new_U = basis @ vectors[:, -rank:]
    new_s2 = values[-rank:]
    return new_U, new_s2


def svd_update(U, s2, x, alpha, rank, tol=1e-12):
    # ONE SVD UPDATE STEP WITH GAIN alpha:
    # computes rank_r((1 - alpha) U diag(s2) U^T + alpha x x^T)
    # discount the retained information from earlier observations,
    # then add the new observation with the sqrt(alpha) forgetting factor
    return svd_rank_one_modify(U, (1 - alpha) * s2, np.sqrt(alpha) * x, +1, rank, tol)


def svd_pca(X, gain, rank=1, tol=1e-12):
    # UPDATES THE SVD OF THE WEIGHTED DATA WITH EACH NEW OBSERVATION, THEN FINDS LEADING DIRECTION
    # gain: fixed alpha in (0, 1], or one alpha_n per observation (array of length n_samples)
    #   alpha_n = 1/n   -> incremental (cumulative) PCA
    #   alpha fixed     -> forgetting PCA
    # rank=1 discards the second component after each update; rank=2 is exact in 2D
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) == 0 or X.shape[1] != 2:
        raise ValueError('X must have shape (n_samples, 2), with at least one sample')
    if not np.isfinite(X).all():
        raise ValueError('X must contain finite values')
    if tol < 0:
        raise ValueError('tol must be nonnegative')
    if rank not in (1, 2):
        raise ValueError('rank must be 1 or 2')

    gains = np.asarray(gain, dtype=float)
    if gains.ndim == 0:
        gains = np.full(len(X), float(gains))
    if gains.shape != (len(X),):
        raise ValueError('gain must be a scalar or have one value per observation')
    if not (np.isfinite(gains).all() and (gains > 0).all() and (gains <= 1).all()):
        raise ValueError('every gain must be in (0,1]')

    # initial retained axes and zero eigenvalues (no information yet)
    U = np.eye(2)[:, :rank]
    s2 = np.zeros(rank)
    q = U[:, -1].copy()
    directions = np.zeros((len(X), 2))

    for index in range(len(X)):
        U, s2 = svd_update(U, s2, X[index], gains[index], rank, tol)
        new_q = U[:, -1]
        # q and -q describe the same axis; keep the sign closest to the previous q
        if np.dot(new_q, q) < 0:
            new_q = -new_q
        q = new_q.copy()
        directions[index] = q

    # reconstruct the retained covariance approximation
    covariance = (U * s2) @ U.T
    return directions, covariance


def rank1_svd_pca(X, alpha, tol=1e-12):
    # RANK-1 SVD PCA WITH FIXED FORGETTING: OLDER INFO IS DISCOUNTED USING THE FIXED GAIN
    # A_n = rank_1((1 - alpha) A_{n-1} + alpha x_n x_n^T)
    if not 0 < alpha <= 1:
        raise ValueError('alpha must be in (0,1]')
    return svd_pca(X, alpha, rank=1, tol=tol)


def _check_window_input(X, window_size):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) == 0 or X.shape[1] != 2:
        raise ValueError('X must have shape (n_samples, 2), with at least one sample')
    if not np.isfinite(X).all():
        raise ValueError('X must contain finite values')
    if int(window_size) != window_size or window_size < 1:
        raise ValueError('window_size must be a positive integer')
    return X, int(window_size)


def sliding_window_cov_pca(X, window_size, recompute_every=None, beta=1.0):
    # SLIDING-WINDOW PCA WITH COVARIANCE UPDATE/DOWNDATE: SAME RESULT AS sliding_window_pca
    # keeps the unnormalized sum S = sum of beta^age x x^T over the window and changes it by two terms per step:
    #   S_n = beta S_{n-1} + x_n x_n^T - beta^w x_{n-w} x_{n-w}^T      cost independent of window_size
    # beta = 1 is the rectangular window; beta < 1 is the truncated exponential window of
    # Badeau, David, Richard (2005), eq. (2)-(3): forgetting inside the window, and a hard cut after w samples
    # the estimate is S divided by the sum of its weights, so the weights of the window sum to one
    # before the window fills, all observations so far are used (as in sliding_window_pca)
    # recompute_every: optionally rebuild S from the window every k steps, to remove rounding drift
    X, window_size = _check_window_input(X, window_size)
    if recompute_every is not None and (int(recompute_every) != recompute_every or recompute_every < 1):
        raise ValueError('recompute_every must be None or a positive integer')
    if not 0 < beta <= 1:
        raise ValueError('beta must be in (0, 1]')

    S = np.zeros((2, 2))
    weight_sum = 0.0
    cut_weight = beta ** window_size   # weight the oldest sample would have when it leaves
    q = np.array([1.0, 0.0])
    directions = np.zeros((len(X), 2))
    for index in range(len(X)):
        x = X[index]
        # update: the older samples lose a factor beta, the new observation enters with weight one
        S = beta * S + np.outer(x, x)
        weight_sum = beta * weight_sum + 1.0
        # downdate: the oldest observation leaves once the window is full
        # (X[index - window_size] is a past observation; a stream would keep it in a buffer)
        if index >= window_size:
            x_old = X[index - window_size]
            S -= cut_weight * np.outer(x_old, x_old)
            weight_sum -= cut_weight
        count = min(index + 1, window_size)

        if recompute_every is not None and (index + 1) % recompute_every == 0:
            window = X[index + 1 - count:index + 1]
            weights = beta ** np.arange(count - 1, -1, -1)
            S = (window * weights[:, None]).T @ window
            weight_sum = weights.sum()

        covariance = S / weight_sum
        q = leading_direction(covariance, q)
        directions[index] = q
    return directions, covariance


def sliding_window_svd_pca(X, window_size, rank=1, tol=1e-12):
    # SLIDING-WINDOW PCA WITH SVD UPDATE/DOWNDATE (rank-one identity of Brand, 2006)
    # unlike Brand's downdate, which uses the stored row of the right singular vectors V of the leaving
    # column, this covariance-form version keeps no V and removes x x^T in the current basis
    # retained memory U diag(s2) U^T approximates S = sum of x x^T over the window
    #   update:   rank_r(S + x_n x_n^T)
    #   downdate: rank_r(S - x_{n-w} x_{n-w}^T)
    # rank=2 is exact in 2D (used in the tests); rank=1 discards the perpendicular part after each
    # modification, so the downdate removes information that was partly never stored: it is approximate
    X, window_size = _check_window_input(X, window_size)
    if rank not in (1, 2):
        raise ValueError('rank must be 1 or 2')
    if tol < 0:
        raise ValueError('tol must be nonnegative')

    U = np.eye(2)[:, :rank]
    s2 = np.zeros(rank)
    q = U[:, -1].copy()
    directions = np.zeros((len(X), 2))
    for index in range(len(X)):
        # update: the new observation enters
        U, s2 = svd_rank_one_modify(U, s2, X[index], +1, rank, tol)
        # downdate: the oldest observation leaves once the window is full
        if index >= window_size:
            U, s2 = svd_rank_one_modify(U, s2, X[index - window_size], -1, rank, tol)
            # a truncated downdate can push retained eigenvalues slightly below zero
            s2 = np.maximum(s2, 0.0)

        new_q = U[:, -1]
        # q and -q describe the same axis; keep the sign closest to the previous q
        if np.dot(new_q, q) < 0:
            new_q = -new_q
        q = new_q.copy()
        directions[index] = q

    count = min(len(X), window_size)
    covariance = (U * s2) @ U.T / count
    return directions, covariance


TW_API_RESCALE_LOG = 50.0   # see tw_api_pca: largest |log| of the running rescaling product before folding it


def tw_api_pca(X, window_size, beta=1.0):
    # SLIDING-WINDOW PCA WITH A DIRECT UPDATE OF THE DIRECTION: TW-API, rank one
    # Badeau, David, Richard (2005), Fast approximated power iteration subspace tracking,
    # IEEE Trans. Signal Process. 53(8), Table II, written out for one direction (r = 1) and real data.
    # window: C_n = beta C_{n-1} + x_n x_n^T - beta^w x_{n-w} x_{n-w}^T (beta = 1: rectangular window)
    # each step is one approximated power iteration q <- C_n q / ||C_n q||, without forming C_n:
    #   y = q.x_new, v = q.x_old                current coordinates of the entering and leaving samples
    #   v_hat                                   coordinate of the leaving sample stored when it entered
    #   e_new = x_new - y q, e_old = x_old - v q   parts perpendicular to q
    #   z ~ 1 / (q^T S q)                       inverse energy of the window along q
    #   g = (z y, -beta^w z v_hat) / (beta + z y^2 - beta^w z v_hat v)
    #   q <- (q + g1 e_new + g2 e_old) * theta  theta renormalizes q exactly
    #   z <- z (1 - g1 y - g2 v) / beta         matrix inversion lemma
    #   v_hat <- theta v_hat                    stored coordinates follow the basis
    # removing the leaving sample with its stored coordinate keeps the bookkeeping consistent with what
    # was added; this is the difference from sliding_window_svd_pca (rank 1), which reprojects it
    # initialization as in the paper: q = first axis, z = 1
    # the approximation assumes that the window matrix changes little at each step (the paper uses w = 120);
    # for very short windows the downdate can remove more energy along q than z accounts for, and the
    # recursion breaks down: this is reported with a FloatingPointError instead of returning invalid directions
    # returns the directions and the final estimate of the leading eigenvalue of the normalized window
    X, window_size = _check_window_input(X, window_size)
    if not 0 < beta <= 1:
        raise ValueError('beta must be in (0, 1]')

    cut_weight = beta ** window_size
    q = np.array([1.0, 0.0])
    z = 1.0
    weight_sum = 0.0
    # the factor theta multiplies every stored coordinate at every step: store each coordinate
    # divided by the running product (kept as a logarithm), so the rescaling costs O(1)
    # theta <= 1, so log_scale only decreases and exp(-log_scale) would overflow on long streams
    # (after about 1e5 steps for w = 5, 6e7 for w = 99); when it passes TW_API_RESCALE_LOG the product
    # is folded into the stored coordinates and reset to zero. This costs O(w) once in a long while
    # and changes the results only by rounding.
    stored = np.zeros(window_size)
    log_scale = 0.0
    directions = np.zeros((len(X), 2))
    for index in range(len(X)):
        x = X[index]
        slot = index % window_size
        y = q @ x
        e_new = x - y * q
        if index >= window_size:
            x_old = X[index - window_size]
            v = q @ x_old
            v_hat = stored[slot] * np.exp(log_scale)
            e_old = x_old - v * q
            weight_sum = beta * weight_sum + 1.0 - cut_weight
        else:
            v = v_hat = 0.0
            e_old = np.zeros(2)
            weight_sum = beta * weight_sum + 1.0
        # the entering sample is stored with its current coordinate
        stored[slot] = y / np.exp(log_scale)

        denominator = beta + z * (y * y - cut_weight * v_hat * v)
        if not denominator > 0:
            raise FloatingPointError('TW-API gain is not defined at observation %d' % (index + 1))
        g_new = z * y / denominator
        g_old = -cut_weight * z * v_hat / denominator

        step = g_new * e_new + g_old * e_old
        theta = 1.0 / np.sqrt(1.0 + step @ step)
        z = z * (1.0 - g_new * y - g_old * v) / beta
        if not z > 0:
            raise FloatingPointError('TW-API energy estimate is not positive at observation %d' % (index + 1))
        q = (q + step) * theta
        log_scale += np.log(theta)
        if log_scale < -TW_API_RESCALE_LOG:
            stored *= np.exp(log_scale)
            log_scale = 0.0
        directions[index] = q

    leading_eigenvalue = 1.0 / (z * weight_sum)
    return directions, leading_eigenvalue


def oja_pca(X, gain):
    # OJA PCA: DIRECT UPDATE OF THE LEADING DIRECTION, WITHOUT A COVARIANCE MATRIX
    # normalized Oja update with controlled gain, eq. (2)-(3) of the thesis proposal:
    #   q_tilde = q_n + alpha_n (I - q_n q_n^T) x x^T q_n = q_n + alpha_n y (x - y q_n),   y = q_n^T x
    #   q_{n+1} = q_tilde / ||q_tilde||
    # (x - y q_n) is perpendicular to q_n, so ||q_tilde|| >= 1 and the normalization is always defined
    # gain: a scalar (fixed gain) or one value per observation (a gain sequence, e.g. chosen by a controller)
    # NOTE: the same alpha does not give the same memory as forgetting_pca: the step multiplies x x^T, so it
    # grows with the variance of the data. Scaled by the spectral gap, alpha_oja = alpha / (lam1 - lam2) gives
    # about the same stationary noise as forgetting_pca with alpha (checked in notebooks/03_oja.ipynb, Section 2)
    # returns the directions and the innovation r_{n+1} = ||(I - q_n q_n^T) x_{n+1}||^2 of eq. (4), computed
    # with the direction available BEFORE the observation is used (it is observable online)
    # known zero population mean, as for the other methods; initial direction = first axis
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) == 0 or X.shape[1] != 2:
        raise ValueError('X must have shape (n_samples, 2), with at least one sample')
    if not np.isfinite(X).all():
        raise ValueError('X must contain finite values')
    gains = np.asarray(gain, dtype=float)
    if gains.ndim == 0:
        gains = np.full(len(X), float(gains))
    if gains.shape != (len(X),):
        raise ValueError('gain must be a scalar or have one value per observation')
    if not (np.isfinite(gains).all() and (gains > 0).all()):
        raise ValueError('every gain must be positive and finite')

    q = np.array([1.0, 0.0])
    directions = np.zeros((len(X), 2))
    innovations = np.zeros(len(X))
    for index in range(len(X)):
        x = X[index]
        y = q @ x
        residual = x - y * q          # (I - q q^T) x, the part of x that q does not explain
        innovations[index] = residual @ residual
        q_tilde = q + gains[index] * y * residual
        q = q_tilde / np.linalg.norm(q_tilde)
        directions[index] = q
    return directions, innovations


def adaptive_gain_pca(
    X, calibration_size=200, calibration_burnin=50,
    beta_initial=0.98, beta_min=0.95, beta_max=0.995,
    k=0.3961, power=1.0
):
    # UPDATES ZERO MEAN COVARIANCE MATRIX WITH EACH NEW OBSERVATION, THEN FINDS LEADING DIRECTION
    # THE FORGETTING FACTOR IS ADJUSTED ADAPTIVELY BASED ON THE MAGNITUDE OF CHANGE IN THE COVARIANCE MATRIX, COMPARED TO A CALIBRATION PERIOD
    # DURING THE INITIAL CALBIRATION PERIOD, IT USES A FIXED FORGETTING FACTOR TO ESTIMATE THE NORMAL MAGNITUDE OF CHANGE IN THE COVARIANCE MATRIX
    # THEN AFTER THE CALIBRATION PERIOD, IT ADJUSTS THE FORGETTING FACTOR BASED ON THE RATIO OF THE CURRENT CHANGE TO THE NORMAL CHANGE
    
    # STRONG ASSUMPTION: we actually do have a calibration period where the data is stationary
    
    if not 0 <= calibration_burnin < calibration_size < len(X):
        raise ValueError('Require 0 <= burnin < calibration_size < len(X)')
    if not 0 <= beta_min <= beta_initial <= beta_max < 1:
        raise ValueError('Require 0 <= beta_min <= beta_initial <= beta_max < 1')
    if k <= 0 or power <= 0:
        raise ValueError('k and power must be positive')

    covariance = np.zeros((2, 2))
    directions = np.zeros((len(X), 2))
    gain_history = np.zeros(len(X))
    q = np.array([1.0, 0.0])

    calibration_changes = []
    previous_change = 0.0

    for index in range(len(X)):
        if index < calibration_size:
            # use a fixed forgetting factor while measuring normal fluctuations.
            beta = beta_initial
        else:
            # Ayech equation (17), applied to our covariance estimate
            # only information from previous observations is used
            ratio = previous_change / max(normal_change, 1e-12)
            beta = beta_min + (beta_max - beta_min) * np.exp(
                -k * ratio**power
            )

        previous_covariance = covariance.copy()
        x = X[index]

        # known zero population mean: outer(x, x) contributes to covariance.
        covariance = beta * covariance + (1 - beta) * np.outer(x, x)

        # measure the change for choosing the next observation's gain.
        previous_change = np.linalg.norm(
            covariance - previous_covariance, ord='fro'
        )

        # exclude early initialization changes from the calibration reference.
        if calibration_burnin <= index < calibration_size:
            calibration_changes.append(previous_change)

        if index == calibration_size - 1:
            normal_change = np.mean(calibration_changes)

        q = leading_direction(covariance, q)
        directions[index] = q
        gain_history[index] = 1 - beta

    return directions, covariance, gain_history
