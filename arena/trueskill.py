"""Role-specific TrueSkill-style ratings for Arena matches."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class TrueSkillRating:
    mu: float = 25.0
    sigma: float = 25.0 / 3.0

    @property
    def conservative(self) -> float:
        return self.mu - 3.0 * self.sigma


def _pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _update_win(
    winner: TrueSkillRating,
    loser: TrueSkillRating,
    *,
    beta: float,
    tau: float,
) -> tuple[TrueSkillRating, TrueSkillRating]:
    winner_sigma = math.sqrt(winner.sigma * winner.sigma + tau * tau)
    loser_sigma = math.sqrt(loser.sigma * loser.sigma + tau * tau)
    c = math.sqrt(2.0 * beta * beta + winner_sigma * winner_sigma + loser_sigma * loser_sigma)
    t = (winner.mu - loser.mu) / c
    denom = max(_cdf(t), 1e-12)
    v = _pdf(t) / denom
    w = v * (v + t)

    winner_mu = winner.mu + (winner_sigma * winner_sigma / c) * v
    loser_mu = loser.mu - (loser_sigma * loser_sigma / c) * v
    winner_var = winner_sigma * winner_sigma * max(
        1e-9,
        1.0 - (winner_sigma * winner_sigma / (c * c)) * w,
    )
    loser_var = loser_sigma * loser_sigma * max(
        1e-9,
        1.0 - (loser_sigma * loser_sigma / (c * c)) * w,
    )
    return (
        TrueSkillRating(mu=winner_mu, sigma=math.sqrt(winner_var)),
        TrueSkillRating(mu=loser_mu, sigma=math.sqrt(loser_var)),
    )


def _rating_dict(rating: TrueSkillRating) -> dict[str, float]:
    return {
        "mu": round(rating.mu, 4),
        "sigma": round(rating.sigma, 4),
        "skill": round(rating.conservative, 4),
    }


def compute_role_trueskill(
    matches: list[dict[str, Any]],
    *,
    mu: float = 25.0,
    sigma: float = 25.0 / 3.0,
    beta: float = 25.0 / 6.0,
    tau: float = 25.0 / 300.0,
    draw_threshold: float = 0.0,
) -> dict[str, Any]:
    """Compute separate detective and culprit TrueSkill ratings.

    Arena payoffs are continuous. TrueSkill itself is ordinal, so each episode
    is converted into a role win/loss by comparing detective payoff against the
    symmetric 0.5 split. Matches inside ``draw_threshold`` of 0.5 are treated
    as draws and leave ratings unchanged.
    """
    detective_ratings: defaultdict[str, TrueSkillRating] = defaultdict(
        lambda: TrueSkillRating(mu=mu, sigma=sigma)
    )
    culprit_ratings: defaultdict[str, TrueSkillRating] = defaultdict(
        lambda: TrueSkillRating(mu=mu, sigma=sigma)
    )

    ordered = sorted(
        matches,
        key=lambda m: (
            str(m.get("level", "")),
            int(m.get("seed", 0)),
            m.get("match_id", ""),
        ),
    )
    for match in ordered:
        if match.get("error"):
            continue
        d_name = match.get("detective", {}).get("name", "unknown")
        c_name = match.get("culprit", {}).get("name", "unknown")
        detective_rating = detective_ratings[d_name]
        culprit_rating = culprit_ratings[c_name]
        payoff = max(0.0, min(1.0, float(match.get("detective_payoff", 0.0))))

        if abs(payoff - 0.5) <= draw_threshold:
            continue
        if payoff > 0.5:
            detective_rating, culprit_rating = _update_win(
                detective_rating,
                culprit_rating,
                beta=beta,
                tau=tau,
            )
        else:
            culprit_rating, detective_rating = _update_win(
                culprit_rating,
                detective_rating,
                beta=beta,
                tau=tau,
            )
        detective_ratings[d_name] = detective_rating
        culprit_ratings[c_name] = culprit_rating

    return {
        "system": "trueskill",
        "params": {
            "mu": mu,
            "sigma": sigma,
            "beta": beta,
            "tau": tau,
            "draw_threshold": draw_threshold,
        },
        "detective": {
            name: _rating_dict(rating)
            for name, rating in sorted(detective_ratings.items())
        },
        "culprit": {
            name: _rating_dict(rating)
            for name, rating in sorted(culprit_ratings.items())
        },
    }
