"""Values printed in the source papers, transcribed for comparison.

Longstaff and Schwartz (2001), Table 1, p. 127: American-style puts with
strike 40 and rate 0.06, exercisable 50 times per year. Read from the text
layer of the copy at
https://www.anderson.ucla.edu/documents/areas/fac/finance/least_squares.pdf

Cox, Ross and Rubinstein (1979), Table 2, p. 259: European calls with S = 40
and r = 1.05 (one plus the annual rate), expiring in one, four and seven
months, by binomial tree with n = 5 and n = 20 and by the Black-Scholes
formula. Read from a scan of the journal pages.
"""

from __future__ import annotations

from dataclasses import dataclass

LS_STRIKE = 40.0
LS_RATE = 0.06
LS_EXERCISE_DATES_PER_YEAR = 50
LS_PATHS = 100_000


@dataclass(frozen=True)
class LsRow:
    """One row of Longstaff and Schwartz, Table 1."""

    spot: float
    vol: float
    maturity: float
    fd_american: float
    european: float
    sim_american: float
    sim_se: float


LS_TABLE_1: tuple[LsRow, ...] = (
    LsRow(36, 0.20, 1, 4.478, 3.844, 4.472, 0.010),
    LsRow(36, 0.20, 2, 4.840, 3.763, 4.821, 0.012),
    LsRow(36, 0.40, 1, 7.101, 6.711, 7.091, 0.020),
    LsRow(36, 0.40, 2, 8.508, 7.700, 8.488, 0.024),
    LsRow(38, 0.20, 1, 3.250, 2.852, 3.244, 0.009),
    LsRow(38, 0.20, 2, 3.745, 2.991, 3.735, 0.011),
    LsRow(38, 0.40, 1, 6.148, 5.834, 6.139, 0.019),
    LsRow(38, 0.40, 2, 7.670, 6.979, 7.669, 0.022),
    LsRow(40, 0.20, 1, 2.314, 2.066, 2.313, 0.009),
    LsRow(40, 0.20, 2, 2.885, 2.356, 2.879, 0.010),
    LsRow(40, 0.40, 1, 5.312, 5.060, 5.308, 0.018),
    LsRow(40, 0.40, 2, 6.920, 6.326, 6.921, 0.022),
    LsRow(42, 0.20, 1, 1.617, 1.465, 1.617, 0.007),
    LsRow(42, 0.20, 2, 2.212, 1.841, 2.206, 0.010),
    LsRow(42, 0.40, 1, 4.582, 4.379, 4.588, 0.017),
    LsRow(42, 0.40, 2, 6.248, 5.736, 6.243, 0.021),
    LsRow(44, 0.20, 1, 1.110, 1.017, 1.118, 0.007),
    LsRow(44, 0.20, 2, 1.690, 1.429, 1.675, 0.009),
    LsRow(44, 0.40, 1, 3.948, 3.783, 3.957, 0.017),
    LsRow(44, 0.40, 2, 5.647, 5.202, 5.622, 0.021),
)

CRR_SPOT = 40.0
CRR_ONE_PLUS_RATE = 1.05
CRR_MONTHS: tuple[int, ...] = (1, 4, 7)

# (vol, strike) -> values for n = 5, n = 20 and n = infinity (Black-Scholes),
# each a triple over the three expiries in CRR_MONTHS.
CRR_TABLE_2: dict[tuple[float, float], dict[str, tuple[float, float, float]]] = {
    (0.2, 35.0): {"5": (5.14, 5.77, 6.45), "20": (5.15, 5.77, 6.39), "inf": (5.15, 5.76, 6.40)},
    (0.2, 40.0): {"5": (1.05, 2.26, 3.12), "20": (0.99, 2.14, 2.97), "inf": (1.00, 2.17, 3.00)},
    (0.2, 45.0): {"5": (0.02, 0.54, 1.15), "20": (0.02, 0.51, 1.11), "inf": (0.02, 0.51, 1.10)},
    (0.3, 35.0): {"5": (5.21, 6.30, 7.15), "20": (5.22, 6.26, 7.19), "inf": (5.22, 6.25, 7.17)},
    (0.3, 40.0): {"5": (1.53, 3.21, 4.36), "20": (1.44, 3.04, 4.14), "inf": (1.46, 3.07, 4.19)},
    (0.3, 45.0): {"5": (0.11, 1.28, 2.12), "20": (0.15, 1.28, 2.23), "inf": (0.16, 1.25, 2.24)},
    (0.4, 35.0): {"5": (5.40, 6.87, 7.92), "20": (5.39, 6.91, 8.05), "inf": (5.39, 6.89, 8.09)},
    (0.4, 40.0): {"5": (2.01, 4.16, 5.61), "20": (1.90, 3.93, 5.31), "inf": (1.92, 3.98, 5.37)},
    (0.4, 45.0): {"5": (0.46, 1.99, 3.30), "20": (0.42, 2.09, 3.42), "inf": (0.42, 2.10, 3.43)},
}
