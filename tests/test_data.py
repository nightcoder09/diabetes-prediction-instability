import numpy as np
import pandas as pd
import pytest

from ehai.data import (FEATURES, TARGET, DataValidationError, canonicalise, content_hash, duplicate_report,
                       label_mapping_check, validate_schema)


def test_schema_accepts_valid(synth):
    rep = validate_schema(synth)
    assert rep["n_missing_cells"] == 0


def test_schema_rejects_out_of_range(synth):
    bad = synth.copy()
    bad.loc[0, "BMI"] = 150
    with pytest.raises(DataValidationError):
        validate_schema(bad)


def test_schema_rejects_missing_column(synth):
    with pytest.raises(DataValidationError):
        validate_schema(synth.drop(columns=["Income"]))


def test_schema_rejects_non_integer(synth):
    bad = synth.astype(float).copy()
    bad.loc[3, "GenHlth"] = 2.5
    with pytest.raises(DataValidationError):
        validate_schema(bad)


def test_canonical_hash_is_row_order_invariant(synth):
    shuffled = synth.sample(frac=1.0, random_state=3)
    assert content_hash(synth) == content_hash(shuffled)
    assert canonicalise(synth).equals(canonicalise(shuffled))


def test_canonical_hash_detects_change(synth):
    other = synth.copy()
    other.loc[10, TARGET] = 1 - other.loc[10, TARGET]
    assert content_hash(synth) != content_hash(other)


def test_label_mapping_check_identifies_prediabetes_class(synth):
    three = synth.rename(columns={TARGET: "Diabetes_012"}).copy()
    three["Diabetes_012"] = np.where(synth[TARGET] == 1, 2, 0)
    three.loc[three.index[:50], "Diabetes_012"] = np.where(synth[TARGET].iloc[:50] == 0, 1, 2)
    rep = label_mapping_check(synth, three)
    assert rep["rows_aligned"] and rep["prediabetes_in_binary_class"] == 0


def test_duplicate_report_counts():
    df = pd.DataFrame({f: [1, 1, 1] for f in FEATURES})
    df[TARGET] = [0, 0, 1]
    rep = duplicate_report(df)
    assert rep["duplicate_rows"] == 1
    assert rep["duplicate_feature_vectors"] == 2
    assert rep["feature_vectors_with_conflicting_labels"] == 1
