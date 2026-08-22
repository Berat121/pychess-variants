"""
glicko2
~~~~~~~

The Glicko2 rating system.

:copyright: (c) 2012 by Heungsub Lee
:Modified by Bajusz Tamás
:license: BSD, see LICENSE for more details.
"""

import math
from calendar import timegm
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

from typing_defs import PerfEntry, PerfMap

#: The actual score for win
WIN = 1.0
#: The actual score for draw
DRAW = 0.5
#: The actual score for loss
LOSS = 0.0

# http://www.glicko.net/glicko/glicko2.pdf
MU = 1500
PHI = 350
SIGMA = 0.06
TAU = 0.75
EPSILON = 0.000001

MIN_MU = 600
MIN_PHI = 30
MAX_SIGMA = 0.1
PROVISIONAL_PHI = 110

# Precomputed constants
_SECONDS_PER_RATING_PERIOD = 60.0 * 60.0 * 24.0 * 4.665
_PI_SQ = math.pi * math.pi
_RD_CAP = 350.0 / 173.7178
_3_OVER_PI_SQ = 3.0 / _PI_SQ


class Rating:
    __slots__ = "ltime", "mu", "phi", "sigma"

    def __init__(
        self,
        mu: float = MU,
        phi: float = PHI,
        sigma: float = SIGMA,
        ltime: datetime | None = None,
    ):
        self.mu = mu
        self.phi = phi
        self.sigma = sigma
        self.ltime = ltime

    @property
    def rating_prov(self) -> tuple[int, str]:
        return (int(round(self.mu, 0)), "?" if self.phi > PROVISIONAL_PHI else "")

    def __repr__(self) -> str:
        return "(mu=%.3f, phi=%.3f, sigma=%.3f, ltime=%s)" % (
            self.mu,
            self.phi,
            self.sigma,
            self.ltime,
        )


def pre_rating_RD(phi: float, sigma: float, ltime: datetime) -> float:
    """
    Calculates the player's rating deviation for the beginning of a rating period.

    4.665 days is the length of a "baseline" rating period used by Lichess,
    which is essentially arbitrary but calibrated so a typical player's RD
    goes from 60 to 110 in a year.
    """
    now_ts = timegm(datetime.now(UTC).timetuple())
    t = (now_ts - timegm(ltime.timetuple())) / _SECONDS_PER_RATING_PERIOD
    if t < 1.0:
        t = 1.0

    ret = math.sqrt(phi * phi + t * sigma * sigma)
    if ret > _RD_CAP:
        ret = _RD_CAP
    return ret


class Glicko2:
    __slots__ = "epsilon", "mu", "phi", "sigma", "tau"

    def __init__(
        self,
        mu: float = MU,
        phi: float = PHI,
        sigma: float = SIGMA,
        tau: float = TAU,
        epsilon: float = EPSILON,
    ):
        self.mu = mu
        self.phi = phi
        self.sigma = sigma
        self.tau = tau
        self.epsilon = epsilon

    def create_rating(
        self,
        mu: float | None = None,
        phi: float | None = None,
        sigma: float | None = None,
        ltime: datetime | None = None,
    ) -> Rating:
        if mu is None:
            mu = self.mu
        if phi is None:
            phi = self.phi
        if sigma is None:
            sigma = self.sigma
        if ltime is None:
            ltime = datetime.now(UTC)
        return Rating(mu, phi, sigma, ltime)

    def scale_down(self, rating: Rating, ratio: float = 173.7178) -> Rating:
        mu = (rating.mu - self.mu) / ratio
        phi = rating.phi / ratio
        return self.create_rating(mu, phi, rating.sigma, rating.ltime)

    def scale_up(self, rating: Rating, ratio: float = 173.7178) -> Rating:
        mu = rating.mu * ratio + self.mu
        phi = rating.phi * ratio
        sigma = rating.sigma

        if mu < MIN_MU:
            mu = MIN_MU
        if phi < MIN_PHI:
            phi = MIN_PHI
        if sigma > MAX_SIGMA:
            sigma = MAX_SIGMA

        return self.create_rating(mu, phi, sigma, rating.ltime)

    @staticmethod
    def reduce_impact(rating: Rating) -> float:
        """The original form is `g(RD)`. This function reduces the impact of
        games as a function of an opponent's RD.
        """
        phi = rating.phi
        return 1.0 / math.sqrt(1.0 + _3_OVER_PI_SQ * phi * phi)

    @staticmethod
    def expect_score(rating: Rating, other_rating: Rating, impact: float) -> float:
        return 1.0 / (1.0 + math.exp(-impact * (rating.mu - other_rating.mu)))

    def determine_sigma(self, rating: Rating, difference: float, variance: float) -> float:
        """Determines new sigma."""
        phi = rating.phi
        phi2 = phi * phi
        diff2 = difference * difference
        phi2_plus_var = phi2 + variance
        alpha = 2.0 * math.log(rating.sigma)

        tau = self.tau
        tau2 = tau * tau
        inv_tau2 = 1.0 / tau2

        def f(x: float) -> float:
            """This function is twice the conditional log-posterior density of
            phi, and is the optimality criterion.
            """
            ex = math.exp(x)
            tmp = phi2_plus_var + ex
            a = ex * (diff2 - tmp) / (2.0 * tmp * tmp)
            b = (x - alpha) * inv_tau2
            return a - b

        a = alpha
        if diff2 > phi2_plus_var:
            b = math.log(diff2 - phi2_plus_var)
        else:
            k = 1
            while f(alpha - k * tau) < 0:
                k += 1
            b = alpha - k * tau

        f_a, f_b = f(a), f(b)

        while abs(b - a) > self.epsilon:
            c = a + (a - b) * f_a / (f_b - f_a)
            f_c = f(c)
            if f_c * f_b < 0:
                a, f_a = b, f_b
            else:
                f_a /= 2.0
            b, f_b = c, f_c

        return math.exp(a / 2.0)

    def rate(self, rating: Rating, series: list[tuple[float, Rating]]) -> Rating:
        # Step 2. For each player, convert the rating and RD's onto the
        #         Glicko-2 scale.
        rating = self.scale_down(rating)

        # Step 3. Compute the quantity v. This is the estimated variance of the
        #         team's/player's rating based only on game outcomes.
        # Step 4. Compute the quantity difference, the estimated improvement in
        #         rating by comparing the pre-period rating to the performance
        #         rating based only on game outcomes.
        variance_inv = 0.0
        difference = 0.0

        if not series:
            # If the team didn't play in the series, do only Step 6
            phi_star = pre_rating_RD(rating.phi, rating.sigma, rating.ltime)
            return self.scale_up(Rating(rating.mu, phi_star, rating.sigma, rating.ltime))

        # Fast path for the common case (1vs1).
        if len(series) == 1:
            actual_score, opponent = series[0]
            opponent = self.scale_down(opponent)
            impact = self.reduce_impact(opponent)
            expected_score = self.expect_score(rating, opponent, impact)
            impact2 = impact * impact
            variance_inv = impact2 * expected_score * (1.0 - expected_score)
            variance = 1.0 / variance_inv
            difference = impact * (actual_score - expected_score) * variance
        else:
            for actual_score, opponent in series:
                opponent = self.scale_down(opponent)
                impact = self.reduce_impact(opponent)
                expected_score = self.expect_score(rating, opponent, impact)
                impact2 = impact * impact
                variance_inv += impact2 * expected_score * (1.0 - expected_score)
                difference += impact * (actual_score - expected_score)
            variance = 1.0 / variance_inv
            difference *= variance

        # Step 5. Determine the new value, Sigma', ot the sigma. This
        #         computation requires iteration.
        sigma = self.determine_sigma(rating, difference, variance)

        # Step 6. Update the rating deviation to the new pre-rating period
        #         value, Phi*.
        phi_star = pre_rating_RD(rating.phi, sigma, rating.ltime)

        # Step 7. Update the rating and RD to the new values, Mu' and Phi'.
        phi_star_sq = phi_star * phi_star
        variance_inv = 1.0 / variance
        phi = 1.0 / math.sqrt(1.0 / phi_star_sq + variance_inv)
        mu = rating.mu + phi * phi * difference

        # Step 8. Convert ratings and RD's back to original scale.
        return self.scale_up(Rating(mu, phi, sigma, rating.ltime))

    def rate_1vs1(
        self,
        rating1: Rating,
        rating2: Rating,
        drawn: bool = False,
    ) -> tuple[Rating, Rating]:
        return (
            self.rate(rating1, [(DRAW if drawn else WIN, rating2)]),
            self.rate(rating2, [(DRAW if drawn else LOSS, rating1)]),
        )

    def quality_1vs1(self, rating1: Rating, rating2: Rating) -> float:
        expected_score1 = self.expect_score(rating1, rating2, self.reduce_impact(rating1))
        expected_score2 = self.expect_score(rating2, rating1, self.reduce_impact(rating2))
        expected_score = (expected_score1 + expected_score2) / 2.0
        return 2.0 * (0.5 - abs(0.5 - expected_score))


# Default instance for backward compatibility
gl2 = Glicko2()


def _perf_timestamp(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            return raw.replace(tzinfo=UTC)
        return raw.astimezone(UTC)
    return datetime.now(UTC)


def new_default_perf(ltime: datetime | None = None) -> PerfEntry:
    timestamp = _perf_timestamp(ltime)
    return {
        "gl": {"r": float(MU), "d": float(PHI), "v": float(SIGMA)},
        "la": timestamp,
        "nb": 0,
    }


def perf_entry_with_defaults(perf: Mapping[str, object] | None = None) -> PerfEntry:
    if perf is None:
        return new_default_perf()

    raw_gl = perf.get("gl")
    gl = raw_gl if isinstance(raw_gl, Mapping) else {}

    raw_nb = perf.get("nb", 0)
    if isinstance(raw_nb, bool):
        nb = int(raw_nb)
    elif isinstance(raw_nb, (int, float, str, bytes)):
        try:
            nb = int(raw_nb)
        except ValueError:
            nb = 0
    else:
        nb = 0

    return {
        "gl": {
            "r": float(gl.get("r", MU)),
            "d": float(gl.get("d", PHI)),
            "v": float(gl.get("v", SIGMA)),
        },
        "la": _perf_timestamp(perf.get("la")),
        "nb": nb,
    }


def new_default_perf_map(variants: Iterable[str]) -> PerfMap:
    return {variant: new_default_perf() for variant in variants}


def is_default_perf(perf: PerfEntry) -> bool:
    gl = perf["gl"]
    return (
        perf["nb"] == 0
        and gl["r"] == float(MU)
        and gl["d"] == float(PHI)
        and gl["v"] == float(SIGMA)
    )


def sparse_perf_map(
    variants: Iterable[str],
    perfs: Mapping[str, Mapping[str, object]] | PerfMap | None = None,
) -> PerfMap:
    """Normalize and retain only ratings that carry non-default state."""
    if perfs is None:
        return {}

    allowed_variants = frozenset(variants)
    normalized: PerfMap = {}
    for variant, perf in perfs.items():
        if variant not in allowed_variants:
            continue
        entry = perf_entry_with_defaults(perf if isinstance(perf, Mapping) else None)
        if not is_default_perf(entry):
            normalized[variant] = entry
    return normalized
