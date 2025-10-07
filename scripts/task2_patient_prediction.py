"""Predict the probability that the 10th patient will be urgent.

The script implements the Bayesian update described in the ADA511 assignment.
It accepts a list of probabilities ``p0`` .. ``p10`` describing how likely it is
to observe a specific *count* of urgent patients in any order.  Given the
observations for the first nine patients, the script outputs the conditional
probability that the 10th patient is urgent.
"""

from __future__ import annotations

import argparse
import math
from typing import Sequence


def parse_probability_vector(raw: str) -> list[float]:
    """Parse a comma separated probability vector."""

    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 11:
        raise argparse.ArgumentTypeError(
            "The probability vector must contain exactly 11 comma separated values."
        )
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:  # pragma: no cover - argument parsing guard
        raise argparse.ArgumentTypeError("Probabilities must be numeric values.") from exc

    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("Probabilities must be non-negative numbers.")
    total = sum(values)
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise argparse.ArgumentTypeError(
            "The probabilities must sum to 1.0 (within a tolerance of 1e-9)."
        )
    return values


def parse_sequence(raw: str) -> list[int]:
    """Parse a comma separated sequence of 0/1 integers."""

    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 9:
        raise argparse.ArgumentTypeError(
            "The patient sequence must contain exactly nine comma separated entries."
        )
    sequence: list[int] = []
    for part in parts:
        if part not in {"0", "1"}:
            raise argparse.ArgumentTypeError(
                "Each patient entry must be either 0 (non-urgent) or 1 (urgent)."
            )
        sequence.append(int(part))
    return sequence


def sequence_probability(probabilities: Sequence[float], sequence: Sequence[int]) -> float:
    """Return the probability of observing *sequence* of length 10.

    The function assumes that *probabilities* follows the assignment convention:
    the entry ``probabilities[n]`` gives the probability of seeing ``n`` urgent
    patients in any order.  All permutations with the same number of urgent
    patients are considered equally likely.
    """

    urgent_count = sum(sequence)
    if not 0 <= urgent_count <= 10:
        raise ValueError("A sequence must contain between 0 and 10 urgent patients (inclusive).")

    total_configurations = math.comb(10, urgent_count)
    if total_configurations == 0:
        raise ValueError("Invalid number of urgent patients for a 10 person sequence.")

    return probabilities[urgent_count] / total_configurations


def predict_tenth_patient(probabilities: Sequence[float], observed: Sequence[int]) -> float:
    """Compute the probability that the 10th patient is urgent.

    Parameters
    ----------
    probabilities:
        Iterable containing ``p0`` .. ``p10`` as described in the assignment.
    observed:
        Sequence with nine entries encoding the urgency (1) or non-urgency (0) of
        the observed patients.
    """

    if len(probabilities) != 11:
        raise ValueError("The probability vector must have exactly 11 entries (p0..p10).")
    if len(observed) != 9:
        raise ValueError("Exactly nine patients must be observed to make the prediction.")

    # Evaluate the likelihood of the two possible 10 patient sequences.
    likelihood_last_urgent = sequence_probability(
        probabilities, tuple(observed) + (1,)
    )
    likelihood_last_nonurgent = sequence_probability(
        probabilities, tuple(observed) + (0,)
    )

    denominator = likelihood_last_urgent + likelihood_last_nonurgent
    if denominator == 0:
        raise ValueError(
            "The provided probabilities assign zero mass to all sequences consistent with the observations."
        )

    return likelihood_last_urgent / denominator


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probabilities",
        required=True,
        type=parse_probability_vector,
        help="Comma separated list with the entries p0, p1, ..., p10.",
    )
    parser.add_argument(
        "--sequence",
        required=True,
        type=parse_sequence,
        help="Comma separated sequence describing the first nine patients (0 for non-urgent, 1 for urgent).",
    )
    parser.add_argument(
        "--as-percentage",
        action="store_true",
        help="Format the result as a percentage rather than a decimal probability.",
    )
    return parser.parse_args(argv)


def format_probability(probability: float, as_percentage: bool) -> str:
    if as_percentage:
        return f"{probability * 100:.1f}%"
    return f"{probability:.4f}"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    probability = predict_tenth_patient(args.probabilities, args.sequence)
    print(format_probability(probability, args.as_percentage))


if __name__ == "__main__":
    main()
