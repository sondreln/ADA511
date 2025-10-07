"""Compute conditional distributions from a joint probability table.

This module exposes helper functions and a small command line interface that can
be used to obtain conditional distributions for rows or columns of a joint
probability table.  It works with the CSV files that accompany the ADA511
course material, but it also functions with any table that follows the same
format: the first row contains column names (the first entry is the header for
row names) and the first column contains the row names.

Example
-------
$ python scripts/task1_conditional.py \
    --file data/income/jointp-sex2-income2.csv \
    --condition-axis column \
    --condition-value Male

The command prints the conditional probability distribution for the two income
levels given that the individual is male.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class JointProbabilityTable:
    """Container for a joint probability table.

    The table is stored as a list of rows, each row holding floating point
    probabilities.  Row and column labels are stored as sequences of strings.
    """

    row_labels: Sequence[str]
    column_labels: Sequence[str]
    values: Sequence[Sequence[float]]

    def column(self, column: int) -> List[float]:
        """Return the column at index *column* as a list."""

        return [float(row[column]) for row in self.values]

    def row(self, row_index: int) -> List[float]:
        """Return the row at index *row_index* as a list."""

        return [float(value) for value in self.values[row_index]]


def load_joint_probability_table(path: Path) -> JointProbabilityTable:
    """Load a joint probability table stored as a CSV file.

    Parameters
    ----------
    path:
        Path to the CSV file.  The file must have the first row representing the
        column headers and the first column containing the row labels.
    """

    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration as exc:  # pragma: no cover - guard against empty files
            raise ValueError(f"The file {path} is empty") from exc

        if len(header) < 2:
            raise ValueError(
                "The CSV file must contain at least one column label besides the row name column."
            )

        column_labels = header[1:]
        row_labels: List[str] = []
        values: List[List[float]] = []

        for row in reader:
            if not row:
                continue
            row_labels.append(row[0])
            try:
                values.append([float(value) for value in row[1:]])
            except ValueError as exc:  # pragma: no cover - guard for malformed numbers
                raise ValueError(
                    f"Encountered a non-numeric probability in row {row!r}."
                ) from exc

        if not values:
            raise ValueError("The CSV file does not contain any data rows.")

    # Basic validation of the table shape.
    width = len(column_labels)
    for index, row in enumerate(values):
        if len(row) != width:
            raise ValueError(
                "All rows in the joint probability table must have the same length. "
                f"Row {index} has length {len(row)}, expected {width}."
            )

    return JointProbabilityTable(row_labels=row_labels, column_labels=column_labels, values=values)


def _resolve_index(labels: Sequence[str], value: str) -> int:
    """Resolve *value* as either a 1-based index or a label.

    Parameters
    ----------
    labels:
        Available labels for the axis.  When *value* cannot be interpreted as an
        integer, the function looks for a matching label.
    value:
        Either a string representation of a 1-based index or the label itself.
    """

    try:
        index = int(value) - 1
    except ValueError:
        index = -1
    else:
        if 0 <= index < len(labels):
            return index
        raise ValueError(
            f"Index {value!r} is outside of the valid range 1..{len(labels)}."
        )

    try:
        return labels.index(value)
    except ValueError as exc:
        raise ValueError(
            f"Value {value!r} is not one of the available labels: {', '.join(labels)}"
        ) from exc


def conditional_distribution(
    table: JointProbabilityTable, *, condition_axis: str, condition_value: str
) -> Tuple[List[str], List[float]]:
    """Compute a conditional distribution from a joint probability table.

    Parameters
    ----------
    table:
        The joint probability table.
    condition_axis:
        Either ``"column"`` or ``"row"``.  Specifies which axis is being
        conditioned on.
    condition_value:
        The column or row label (or 1-based index) that should be conditioned on.

    Returns
    -------
    tuple of (labels, probabilities)
        The labels correspond to the axis that is *not* conditioned on.  The
        probabilities represent the conditional distribution for those labels
        given the conditioning value.
    """

    axis = condition_axis.lower()
    if axis not in {"column", "row"}:
        raise ValueError("condition_axis must be either 'column' or 'row'.")

    if axis == "column":
        column_index = _resolve_index(table.column_labels, condition_value)
        raw_values = table.column(column_index)
        target_labels = list(table.row_labels)
    else:
        row_index = _resolve_index(table.row_labels, condition_value)
        raw_values = table.row(row_index)
        target_labels = list(table.column_labels)

    total = sum(raw_values)
    if total <= 0:
        raise ValueError(
            "The selected row or column has a total probability mass that is not strictly positive."
        )

    conditional_probs = [value / total for value in raw_values]
    return target_labels, conditional_probs


def format_distribution(labels: Iterable[str], probabilities: Iterable[float]) -> str:
    """Format a conditional distribution for display."""

    parts = [f"{label}: {prob:.4f}" for label, prob in zip(labels, probabilities)]
    return "[" + ", ".join(parts) + "]"


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Path to the CSV file containing the joint probability table.",
    )
    parser.add_argument(
        "--condition-axis",
        choices=["row", "column"],
        default="column",
        help="Condition on a row or a column (default: column).",
    )
    parser.add_argument(
        "--condition-value",
        required=True,
        help="Label or 1-based index of the row/column to condition on.",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    table = load_joint_probability_table(args.file)
    labels, probabilities = conditional_distribution(
        table, condition_axis=args.condition_axis, condition_value=args.condition_value
    )
    print(format_distribution(labels, probabilities))


if __name__ == "__main__":
    main()
