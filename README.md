# ADA511 Assignment Helpers

This repository contains two small Python utilities that support the ADA511
assignment.

## Task 1 – Conditional distributions from joint tables

`scripts/task1_conditional.py` loads a joint probability table from a CSV file
and returns the conditional distribution for the values in one axis given a
specific entry on the other axis.  The command works with the joint probability
files made available for the course (see the sample data in `data/income/`).

Usage example:

```bash
python scripts/task1_conditional.py \
    --file data/income/jointp-sex2-income2.csv \
    --condition-axis column \
    --condition-value Male
```

The output is formatted as a list of label/probability pairs.  You can use
labels or 1-based indices to identify rows or columns, e.g. `--condition-value 2`
selects the second column in the table.

## Task 2 – Predicting the 10th patient

`scripts/task2_patient_prediction.py` evaluates the probability that the 10th
patient in a 10-patient sequence will be urgent.  The script accepts the
probability vector `p0` .. `p10` (comma separated) and the observed sequence of
nine patients.  It applies the conditional probability formula under the
assumption that all permutations with the same number of urgent patients are
equally likely.

Usage example:

```bash
python scripts/task2_patient_prediction.py \
    --probabilities "0.204,0.013,0.012,0.112,0.127,0.095,0.073,0.209,0.011,0.064,0.08" \
    --sequence "1,1,0,1,1,1,0,0,1" \
    --as-percentage
```

The `--as-percentage` flag optionally formats the result as a percentage.
Without the flag, the probability is displayed as a decimal number.

## Data

The `data/income` directory contains a subset of the ADA511 income joint
probability tables that can be used to test the Task 1 script.
