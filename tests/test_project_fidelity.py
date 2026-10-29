"""The embedded dataset is a typed snapshot, not a JSON number approximation."""
import base64
from datetime import timedelta, timezone
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from dateutil.tz import tzutc

from mlstudio.project_data import pack_frame, unpack_frame


def roundtrip(frame):
    bundle = pack_frame(frame)
    assert bundle["encoding"] == "typed-v4"
    # Exercise the actual disk representation, with strict standard JSON.
    return unpack_frame(json.loads(json.dumps(bundle, allow_nan=False)))


def resign(bundle):
    core = {key: value for key, value in bundle.items() if key != "sha256"}
    bundle["sha256"] = hashlib.sha256(json.dumps(
        core, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    return bundle


def test_uint64_maximum_and_integer_boundaries_are_exact():
    frame = pd.DataFrame({
        "unsigned": np.array([0, 2**63 + 1, 2**64 - 1], dtype="uint64"),
        "signed": np.array([-(2**63), 2**53 + 1, 2**63 - 1], dtype="int64"),
    })
    pd.testing.assert_frame_equal(frame, roundtrip(frame))


@pytest.mark.parametrize("dtype", [
    "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64",
])
def test_nullable_integer_values_dtype_and_mask(dtype):
    limits = np.iinfo(pd.api.types.pandas_dtype(dtype).numpy_dtype)
    frame = pd.DataFrame({"n": pd.Series([limits.min, pd.NA, limits.max], dtype=dtype)})
    restored = roundtrip(frame)
    pd.testing.assert_frame_equal(frame, restored)
    np.testing.assert_array_equal(frame.n.array._mask, restored.n.array._mask)


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float32", "Float64"])
def test_float_bits_infinity_signed_zero_and_nullable_nan(dtype):
    width = 4 if dtype.lower() == "float32" else 8
    uint = np.dtype(f"<u{width}")
    floating = np.dtype(f"<f{width}")
    bits = ([0x80000000, 0x7FC00042, 0x7F800000, 0xFF800000, 0x3F800001]
            if width == 4 else
            [0x8000000000000000, 0x7FF8000000000042, 0x7FF0000000000000,
             0xFFF0000000000000, 0x3FF0000000000001])
    values = np.array(bits, dtype=uint).view(floating)
    if dtype[0].isupper():
        series = pd.Series(pd.arrays.FloatingArray(values, np.array([False]*4 + [True])))
    else:
        series = pd.Series(values)
    frame = pd.DataFrame({"f": series})
    restored = roundtrip(frame)
    pd.testing.assert_frame_equal(frame, restored)
    before = frame.f.array._data if dtype[0].isupper() else frame.f.to_numpy()
    after = restored.f.array._data if dtype[0].isupper() else restored.f.to_numpy()
    np.testing.assert_array_equal(before.view(uint), after.view(uint))
    if dtype[0].isupper():
        assert not restored.f.array._mask[1]  # A valid NaN is distinct from pd.NA.
        np.testing.assert_array_equal(frame.f.array._mask, restored.f.array._mask)


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("timezone", [None, "UTC", "Asia/Dhaka", "America/New_York"])
def test_timestamp_unit_timezone_nat_and_epoch_values(unit, timezone):
    values = pd.DatetimeIndex(["2020-11-01 05:30:00", None, "2020-11-01 06:30:00"])
    if timezone is not None:
        values = values.tz_localize("UTC").tz_convert(timezone)
    frame = pd.DataFrame({"when": values.as_unit(unit)})
    restored = roundtrip(frame)
    pd.testing.assert_frame_equal(frame, restored)
    np.testing.assert_array_equal(frame.when.array.asi8, restored.when.array.asi8)


@pytest.mark.parametrize("categories", [
    pd.Index(["unused", "middle", "first"], dtype="object"),
    pd.Index(np.array([2**64 - 1, 0, 2**63 + 1], dtype="uint64")),
    pd.date_range("2020-01-01", periods=3, tz="Asia/Dhaka").as_unit("us"),
])
def test_categories_keep_unused_domain_dtype_order_and_missing(categories):
    frame = pd.DataFrame({"grade": pd.Categorical.from_codes(
        [2, -1, 1, 2], categories=categories, ordered=True,
    )})
    restored = roundtrip(frame)
    pd.testing.assert_frame_equal(frame, restored)
    pd.testing.assert_index_equal(frame.grade.cat.categories, restored.grade.cat.categories)


@pytest.mark.parametrize("dtype", ["boolean", "bool", "string", "str"])
def test_boolean_and_string_extension_types(dtype):
    values = [True, False] if dtype == "bool" else (
        [True, pd.NA] if dtype == "boolean" else ["nul\x00 and Unicode é", None]
    )
    frame = pd.DataFrame({"value": pd.Series(values, dtype=dtype)})
    pd.testing.assert_frame_equal(frame, roundtrip(frame))


def test_primitive_object_values_remain_typed():
    frame = pd.DataFrame({"mixed": pd.Series(
        ["1", 1, True, None, pd.NA, pd.NaT, -0.0, float("nan")], dtype="object",
    )})
    restored = roundtrip(frame)
    pd.testing.assert_frame_equal(frame, restored)
    assert [type(value) for value in restored.mixed] == [type(value) for value in frame.mixed]
    assert np.signbit(restored.mixed.iloc[6])


def test_provenance_attrs_duplicate_labels_and_normalized_index():
    frame = pd.DataFrame([[1, 2], [3, 4]], columns=["same", "same"], index=[7, 7])
    frame.attrs = {
        "duplicate_headers": ["same"],
        "arff_relation": "quoted relation",
        "arff_schema": [{"name": "same", "kind": "nominal", "values": ["b", "a"]}],
        "temporary_cache": object(),
    }
    restored = roundtrip(frame)
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), restored)
    assert restored.attrs == {key: value for key, value in frame.attrs.items()
                              if key != "temporary_cache"}


@pytest.mark.parametrize("value", [object(), float("nan"), {1: "non-string key"}])
def test_provenance_rejects_non_json_safe_values(value):
    frame = pd.DataFrame({"x": [1]})
    frame.attrs["arff_schema"] = [value]
    with pytest.raises(ValueError, match="metadata|attribute|JSON"):
        pack_frame(frame)


def test_cyclic_provenance_is_rejected_as_unsafe_json():
    cyclic = []
    cyclic.append(cyclic)
    frame = pd.DataFrame({"x": [1]})
    frame.attrs["arff_schema"] = cyclic
    with pytest.raises(ValueError, match="JSON"):
        pack_frame(frame)


@pytest.mark.parametrize("custom", [
    timezone(timedelta(hours=6), "BDT"),
    timezone(timedelta(hours=6), "EST"),
    timezone(timedelta(hours=5, minutes=45, seconds=30)),
    tzutc(),
])
def test_unrepresentable_custom_timezone_is_rejected_before_writing(custom):
    frame = pd.DataFrame({"when": pd.date_range("2020", periods=2, tz=custom)})
    with pytest.raises(ValueError, match="Unsupported timestamp timezone"):
        pack_frame(frame)


def test_standard_fixed_offset_timezone_roundtrips():
    fixed = timezone(timedelta(hours=5, minutes=30))
    frame = pd.DataFrame({"when": pd.date_range("2020", periods=2, tz=fixed)})
    pd.testing.assert_frame_equal(frame, roundtrip(frame))


@pytest.mark.parametrize("series", [
    pd.Series([object()], dtype="object"),
    pd.Series([1 + 2j], dtype="complex128"),
    pd.Series(pd.period_range("2020", periods=1, freq="Y")),
])
def test_unsupported_objects_and_dtypes_fail_clearly(series):
    with pytest.raises(ValueError, match="Unsupported|unsupported"):
        pack_frame(pd.DataFrame({"value": series}))


def legacy_bundle(encoding):
    frame = pd.DataFrame({"x": [1.25, np.nan, np.inf, -np.inf], "name": ["a", "b", "c", "d"]})
    raw = frame.to_json(orient="table", index=False, date_format="iso", double_precision=15)
    if encoding in (None, "typed-v1"):
        bundle = {"data": raw, "sha256": hashlib.sha256(raw.encode()).hexdigest()}
        expected = frame.replace([np.inf, -np.inf], np.nan)
    else:
        bundle = {"data": raw, "infinity": [[2, "x", 1], [3, "x", -1]]}
        if encoding == "typed-v3":
            bundle["floats"] = [{"column": "x", "dtype": "float64", "data":
                                 base64.b64encode(frame.x.to_numpy(dtype="<f8").tobytes()).decode()}]
        bundle["sha256"] = hashlib.sha256(json.dumps(bundle, sort_keys=True).encode()).hexdigest()
        expected = frame
    if encoding is not None:
        bundle["encoding"] = encoding
    return bundle, expected


@pytest.mark.parametrize("encoding", [None, "typed-v1", "typed-v2", "typed-v3"])
def test_legacy_snapshot_readers(encoding):
    bundle, expected = legacy_bundle(encoding)
    pd.testing.assert_frame_equal(expected, unpack_frame(bundle))


def corruption_frame():
    return pd.DataFrame({
        "n": pd.Series([1, pd.NA], dtype="Int64"),
        "when": pd.to_datetime(["2020-01-01", None]),
        "category": pd.Categorical(["a", None], categories=["unused", "a"]),
    })


@pytest.mark.parametrize("mutation", [
    lambda b: b.update(rows=True),
    lambda b: b.update(rows=-1),
    lambda b: b["columns"][0].update(data="%%%"),
    lambda b: b["columns"][0].update(mask=base64.b64encode(b"\0").decode()),
    lambda b: b["columns"][0].update(mask=base64.b64encode(b"\0\2").decode()),
    lambda b: b["columns"][0].update(dtype="object"),
    lambda b: b["columns"][0].pop("kind"),
    lambda b: b["columns"][1].update(unit="D"),
    lambda b: b["columns"][2].update(codes=base64.b64encode(np.array([2, -1], dtype="<i8").tobytes()).decode()),
    lambda b: b["columns"][0].update(name=4),
    lambda b: b["attrs"].update(unrecognized="metadata"),
])
def test_recomputed_checksum_does_not_bypass_structural_validation(mutation):
    bundle = pack_frame(corruption_frame())
    mutation(bundle)
    resign(bundle)
    with pytest.raises(ValueError):
        unpack_frame(bundle)


def test_checksum_covers_dtype_values_and_metadata():
    frame = pd.DataFrame({"x": [1]})
    frame.attrs["duplicate_headers"] = ["x"]
    bundle = pack_frame(frame)
    bundle["attrs"]["duplicate_headers"] = []
    with pytest.raises(ValueError, match="integrity"):
        unpack_frame(bundle)


@pytest.mark.parametrize("encoding", ["typed-v0", "typed-v5", "pickle", 4])
def test_unknown_encoding_is_not_treated_as_legacy(encoding):
    bundle, _ = legacy_bundle(None)
    bundle["encoding"] = encoding
    with pytest.raises(ValueError, match="encoding|Unsupported|unsupported"):
        unpack_frame(bundle)


@pytest.mark.parametrize("bundle", [None, [], {}, {"encoding": "typed-v4"}])
def test_malformed_top_level_is_a_value_error(bundle):
    with pytest.raises(ValueError):
        unpack_frame(bundle)
