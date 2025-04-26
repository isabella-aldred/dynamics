"""A module containing LTI equations for the Clohessy-Wiltshire equations [1].

[1] W. H. CLOHESSY and R. S. WILTSHIRE, “Terminal guidance system for satellite rendezvous,”
    Journal of the Aerospace Sciences, vol. 27, no. 9, pp. 653-658, Sep. 1960. doi:10.2514/8.8704

Module variables include mu values for celestiall bodies listed in
    https://en.wikipedia.org/wiki/Standard_gravitational_parameter with format {CELESTIAL BODY}_MU,
    i.e. EARTH_MU for Earth
"""

import numpy as np
from numpy import typing as npt

# the standard gravitational parameter (mu) of various celestial bodys in m**3 sec**-2
SUN_MU = 1.32712440018e20
MERCURY_MU = 2.20320e13
VENUS_MU = 3.24858593e14
EARTH_MU = 3.986004418e14
MOON_MU = 4.9048695e12
MARS_MU = 4.282837e13
CERES_MU = 6.26325e10
JUPITER_MU = 1.26686534e17
SATURN_MU = 3.7931187e16
URANUS_MU = 5.793939e15
NEPTUNE_MU = 6.836529e15
PLUTO_MU = 8.71e11
ERIS_MU = 1.108e12

# matrix transformation to change A and B from 6d to 4d
_4D_TRANSFORM_A = np.matrix(
    [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]]
)

# matrix transformation to change B from 6d to 4d
_4D_TRANSFORM_B = np.matrix([[1, 0], [0, 1], [0, 0]])


def dt_lti(
    t_s: float, r: float, mu: float = EARTH_MU, four_dim: bool = False
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Discrete-time LTI CWH dynamics with form
        x(t+1) = A*x(t_s) + B*u(t_s)

    Assumptions:
        - two-body dynamics
        - circular orbits
        - impulse control

    Parameters
    ----------
    t_s : float (non-negative)
        sampling period in seconds for discrete time dynamics, i.e. e**{A*t_s}
    r : float (non-negative)
        distance in meters of the orbit from the center of the celestial body
    mu : float, default=EARTH_MU
        the standard gravitational constant in m**3 sec**-2 for a celestial body (see
            https://en.wikipedia.org/wiki/Standard_gravitational_parameter for details)
    four_dim : bool, default=False
        if True, returns a 4-d system with state [x, y, x_dot, y_dot]

    Returns
    -------
    A : np.matrix
        the state transition matrix for sampling period t_s
    B : np.matrix
        the state control matrix for sampling period t_s
    """

    orbit_ang_vel = np.sqrt(mu / r**3)  # rad sec**-2

    sin_nt = np.sin(orbit_ang_vel * t_s)
    cos_nt = np.cos(orbit_ang_vel * t_s)

    # state transformation matrix generation
    A = np.matrix(
        [
            [
                4 - 3 * cos_nt,
                0,
                0,
                (1 / orbit_ang_vel) * sin_nt,
                (2 / orbit_ang_vel) * (1 - cos_nt),
                0,
            ],
            [
                6 * (sin_nt - orbit_ang_vel * t_s),
                1,
                0,
                -(2 / orbit_ang_vel) * (1 - cos_nt),
                (1 / orbit_ang_vel) * (4 * sin_nt - 3 * orbit_ang_vel * t_s),
                0,
            ],
            [
                0,
                0,
                cos_nt,
                0,
                0,
                (1 / orbit_ang_vel) * sin_nt,
            ],
            [
                3 * orbit_ang_vel * sin_nt,
                0,
                0,
                cos_nt,
                2 * sin_nt,
                0,
            ],
            [
                -6 * orbit_ang_vel * (1 - cos_nt),
                0,
                0,
                -2 * sin_nt,
                4 * cos_nt - 3,
                0,
            ],
            [
                0,
                0,
                -orbit_ang_vel * sin_nt,
                0,
                0,
                cos_nt,
            ],
        ]
    )

    # control matrix generation
    B = A * np.concatenate([np.zeros([3, 3]), np.identity(3)])

    # dimensional reduction, if desired
    if four_dim:
        A = _4D_TRANSFORM_A * A * _4D_TRANSFORM_A.T
        B = _4D_TRANSFORM_A * B * _4D_TRANSFORM_B

    return A, B
