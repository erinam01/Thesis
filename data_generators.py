#### GENERATE SYNTHETIC DATA ######

# REGIMES:
# - Stationary: the data is generated from a single distribution.
# - Abrupt: the data is generated from one distribution, then abruptly switches to another.
# - Smooth rotation: the data is generated from a distribution that rotates smoothly over time.
# - Random drift: the direction follows a random walk (eq. 5 of the thesis proposal).

import numpy as np


def generate_stationary(seed, n_samples=2000, lam1=5.0, lam2=0.5, theta0=0.0):
    # direction never changes; all observations are drawn from the same distribution
    rng = np.random.default_rng(seed)
    theta = np.full(n_samples, theta0)

    z = rng.standard_normal((n_samples, 2))
    z[:, 0] = z[:, 0] * np.sqrt(lam1)
    z[:, 1] = z[:, 1] * np.sqrt(lam2)

    # rotate each point (only theta0 for all of them)) --> PROF FESTA COMPUTATION
    c = np.cos(theta)
    s = np.sin(theta)
    x = c * z[:, 0] - s * z[:, 1]
    y = s * z[:, 0] + c * z[:, 1]
    X = np.zeros((n_samples, 2))
    X[:, 0] = x 
    X[:, 1] = y

    truth = np.zeros((n_samples, 2))
    truth[:, 0] = c
    truth[:, 1] = s

    return X, theta, truth


def generate_abrupt(seed, n_samples=2000, lam1=5.0, lam2=0.5, theta0=0.0, theta1=np.pi / 3, switch_step=1000):
    # direction changes abruptly at the switch step; all observations before the switch are drawn from one distribution, 
    # and all observations after the switch are drawn from another
    rng = np.random.default_rng(seed)
    theta = np.full(n_samples, theta0)
    # abrupt change in the angle at the switch step
    theta[switch_step:] = theta1 

    # generate independent coordinates, then scale them to the desired variances.
    z = rng.standard_normal((n_samples, 2))
    z[:, 0] = z[:, 0] * np.sqrt(lam1)
    z[:, 1] = z[:, 1] * np.sqrt(lam2)

    # rotate each point through its assigned angle.
    c = np.cos(theta)
    s = np.sin(theta)
    x = c * z[:, 0] - s * z[:, 1]
    y = s * z[:, 0] + c * z[:, 1]
    X = np.column_stack((x, y))
    truth = np.column_stack((c, s))
    return X, theta, truth

def generate_smooth_rotation(seed, n_samples=2000, lam1=5.0, lam2=0.5, theta0=0.0, omega=0.01, dt=1.0):
    # direction rotates at the constant angular speed omega, starting from theta0:
    #   theta_t = theta0 + omega * dt * t   (no final angle: the rotation never stops)

    rng = np.random.default_rng(seed)
    # smooth rotation from theta0
    theta = theta0 + omega * dt * np.arange(n_samples)

    # generate independent coordinates, then scale them to the desired variances.
    z = rng.standard_normal((n_samples, 2))
    z[:, 0] = z[:, 0] * np.sqrt(lam1)
    z[:, 1] = z[:, 1] * np.sqrt(lam2)

    # rotate each point through its assigned angle.
    c = np.cos(theta)
    s = np.sin(theta)
    x = c * z[:, 0] - s * z[:, 1]
    y = s * z[:, 0] + c * z[:, 1]
    X = np.column_stack((x, y))
    truth = np.column_stack((c, s))
    return X, theta, truth

def generate_drift(seed, n_samples=2000, lam1=5.0, lam2=0.5, theta0=0.0, Q=0.0001, dt=1.0):
    # direction follows a random walk starting from theta0:
    #   theta_t = theta_{t-1} + increment, increments independent N(0, Q * dt)
    rng = np.random.default_rng(seed)

    # generate random rotation increments from a normal distribution with mean 0 and variance Q*dt
    increments = rng.normal(
        loc=0.0, scale=np.sqrt(Q * dt), size=n_samples - 1
    )
    theta = np.zeros(n_samples)
    theta[0] = theta0
    theta[1:] = theta0 + np.cumsum(increments)

    z = rng.standard_normal((n_samples, 2))
    z[:, 0] *= np.sqrt(lam1)
    z[:, 1] *= np.sqrt(lam2)

    c = np.cos(theta)
    s = np.sin(theta)
    x = c * z[:, 0] - s * z[:, 1]
    y = s * z[:, 0] + c * z[:, 1]

    X = np.column_stack((x, y))
    truth = np.column_stack((c, s))
    return X, theta, truth

def covariance_at(theta, lam1, lam2):
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    # rotate the axis-aligned covariance
    return rotation @ np.diag([lam1, lam2]) @ rotation.T
