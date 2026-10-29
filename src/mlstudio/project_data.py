"""Safe typed dataset snapshots, with readers for earlier project encodings.

Version 4 stores numeric buffers without passing through JSON numbers. The index
is intentionally normalized, matching imported datasets and previous snapshots.
Only the import provenance attributes used by ML Studio are persisted.
"""
import base64
import hashlib
import io
import json

import numpy as np
import pandas as pd


_PRIMITIVES = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16",
               "uint32", "uint64", "float16", "float32", "float64"]
_NUMPY_DTYPES = {np.dtype(name).newbyteorder(order).str: np.dtype(name).newbyteorder(order)
                 for name in _PRIMITIVES for order in ("<", ">")}
_NULLABLE_DTYPES = {name: pd.api.types.pandas_dtype(name) for name in (
    "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64",
    "Float32", "Float64", "boolean",
)}
_UNITS = {"s", "ms", "us", "ns"}
_ATTRS = {"duplicate_headers", "arff_relation", "arff_schema"}


def _keys(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("Invalid stored dataset descriptor.")


def _json_safe(value, depth=0):
    if depth > 64:
        raise ValueError("Dataset metadata must contain finite, acyclic JSON-safe values.")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and np.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _json_safe(item, depth + 1)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json_safe(item, depth + 1)
        return
    raise ValueError("Dataset metadata must contain only finite JSON-safe values.")


def _attributes(attrs, *, reading=False):
    if type(attrs) is not dict or (reading and set(attrs) - _ATTRS):
        raise ValueError("Invalid stored dataset metadata attributes.")
    selected = {key: value for key, value in attrs.items() if key in _ATTRS}
    _json_safe(selected)
    if "duplicate_headers" in selected:
        headers = selected["duplicate_headers"]
        if type(headers) is not list or any(type(name) is not str for name in headers):
            raise ValueError("Invalid duplicate-header metadata attribute.")
    if "arff_relation" in selected and type(selected["arff_relation"]) is not str:
        raise ValueError("Invalid ARFF relation metadata attribute.")
    if "arff_schema" in selected and type(selected["arff_schema"]) is not list:
        raise ValueError("Invalid ARFF schema metadata attribute.")
    # Detach nested lists/dicts from the live dataframe without serialization hooks.
    return json.loads(json.dumps(selected, allow_nan=False))


def _digest(core):
    try:
        raw = json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid stored dataset JSON values.") from exc
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _buffer(values, dtype):
    little = np.dtype(dtype).newbyteorder("<")
    return base64.b64encode(np.asarray(values).astype(little, copy=False).tobytes()).decode("ascii")


def _read_buffer(data, dtype, length):
    if type(data) is not str:
        raise ValueError("Invalid stored dataset buffer.")
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid base64 dataset buffer.") from exc
    dtype = np.dtype(dtype)
    if len(raw) != length * dtype.itemsize:
        raise ValueError("Invalid stored dataset buffer length.")
    if dtype.kind == "b" and any(byte > 1 for byte in raw):
        raise ValueError("Invalid stored Boolean dataset values.")
    return np.frombuffer(raw, dtype=dtype.newbyteorder("<")).astype(dtype, copy=True)


def _pack_scalar(value):
    if value is None:
        return ["none"]
    if value is pd.NA:
        return ["na"]
    if value is pd.NaT:
        return ["nat"]
    if type(value) in (str, bool, int):
        return [{str: "str", bool: "bool", int: "int"}[type(value)], value]
    if type(value) is float:
        return ["float", _buffer([value], "<f8")]
    if isinstance(value, np.generic) and value.dtype.str in _NUMPY_DTYPES:
        return ["numpy", value.dtype.str, _buffer([value], value.dtype)]
    raise ValueError(f"Unsupported object value in dataset snapshot: {type(value).__name__}.")


def _read_scalar(value):
    if type(value) is not list or not value or type(value[0]) is not str:
        raise ValueError("Invalid stored object scalar.")
    tag = value[0]
    if len(value) == 1 and tag in {"none", "na", "nat"}:
        return {"none": None, "na": pd.NA, "nat": pd.NaT}[tag]
    scalar_types = {"str": str, "bool": bool, "int": int}
    if len(value) == 2 and tag in scalar_types and type(value[1]) is scalar_types[tag]:
        return value[1]
    if len(value) == 2 and tag == "float":
        return float(_read_buffer(value[1], "<f8", 1)[0])
    if len(value) == 3 and tag == "numpy" and type(value[1]) is str and value[1] in _NUMPY_DTYPES:
        return _read_buffer(value[2], _NUMPY_DTYPES[value[1]], 1)[0]
    raise ValueError("Invalid stored object scalar descriptor.")


def _timezone_name(dtype):
    if not isinstance(dtype, pd.DatetimeTZDtype):
        return None
    name = str(dtype.tz)
    try:
        restored = pd.DatetimeTZDtype(unit=dtype.unit, tz=name)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Unsupported timestamp timezone in dataset snapshot.") from exc
    if restored != dtype:
        raise ValueError("Unsupported timestamp timezone in dataset snapshot.")
    return name


def _pack_column(series, depth=0):
    if depth > 8:
        raise ValueError("Unsupported nested categorical dtype in dataset snapshot.")
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return {"kind": "category", "ordered": dtype.ordered,
                "categories": _pack_column(pd.Series(dtype.categories, dtype=dtype.categories.dtype), depth + 1),
                "category_count": len(dtype.categories),
                "codes": _buffer(series.cat.codes.to_numpy(), "<i8")}
    if isinstance(dtype, pd.DatetimeTZDtype) or (isinstance(dtype, np.dtype) and dtype.kind == "M"):
        unit = dtype.unit if isinstance(dtype, pd.DatetimeTZDtype) else np.datetime_data(dtype)[0]
        if unit not in _UNITS:
            raise ValueError(f"Unsupported timestamp unit in dataset snapshot: {unit}.")
        return {"kind": "datetime", "unit": unit,
                "timezone": _timezone_name(dtype),
                "data": _buffer(series.array.asi8, "<i8")}
    if isinstance(dtype, np.dtype) and dtype.kind == "m":
        unit = np.datetime_data(dtype)[0]
        if unit not in _UNITS:
            raise ValueError(f"Unsupported timedelta unit in dataset snapshot: {unit}.")
        return {"kind": "timedelta", "unit": unit, "data": _buffer(series.array.asi8, "<i8")}
    if isinstance(dtype, pd.StringDtype):
        if dtype.storage not in {"python", "pyarrow"}:
            raise ValueError(f"Unsupported string storage in dataset snapshot: {dtype.storage}.")
        return {"kind": "string", "storage": dtype.storage,
                "na_value": "NA" if dtype.na_value is pd.NA else "nan",
                "values": [None if missing else value for value, missing in zip(series, series.isna())]}
    if str(dtype) in _NULLABLE_DTYPES:
        # These pandas array classes expose separate data/mask buffers. Keeping
        # both also preserves a valid NaN separately from a masked missing value.
        array = series.array
        return {"kind": "nullable", "dtype": str(dtype),
                "data": _buffer(array._data, dtype.numpy_dtype),
                "mask": _buffer(array._mask, "|b1")}
    if isinstance(dtype, np.dtype) and dtype.str in _NUMPY_DTYPES:
        return {"kind": "numpy", "dtype": dtype.str,
                "data": _buffer(series.to_numpy(copy=False), dtype)}
    if isinstance(dtype, np.dtype) and dtype.kind == "O":
        return {"kind": "object", "values": [_pack_scalar(value) for value in series]}
    raise ValueError(f"Unsupported dtype in dataset snapshot: {dtype}.")


def _length(value):
    if type(value) is not int or value < 0:
        raise ValueError("Invalid stored dataset length.")
    return value


def _read_column(block, length, depth=0):
    if depth > 8 or type(block) is not dict or type(block.get("kind")) is not str:
        raise ValueError("Invalid stored dataset column descriptor.")
    kind = block["kind"]
    if kind == "numpy":
        _keys(block, ["kind", "dtype", "data"])
        if type(block["dtype"]) is not str or block["dtype"] not in _NUMPY_DTYPES:
            raise ValueError("Unsupported stored numeric dtype descriptor.")
        return pd.Series(_read_buffer(block["data"], _NUMPY_DTYPES[block["dtype"]], length))
    if kind == "nullable":
        _keys(block, ["kind", "dtype", "data", "mask"])
        if type(block["dtype"]) is not str or block["dtype"] not in _NULLABLE_DTYPES:
            raise ValueError("Unsupported stored nullable dtype descriptor.")
        dtype = _NULLABLE_DTYPES[block["dtype"]]
        data = _read_buffer(block["data"], dtype.numpy_dtype, length)
        mask = _read_buffer(block["mask"], "|b1", length)
        constructor = (pd.arrays.BooleanArray if block["dtype"] == "boolean" else
                       pd.arrays.FloatingArray if block["dtype"].startswith("Float") else
                       pd.arrays.IntegerArray)
        return pd.Series(constructor(data, mask))
    if kind in {"datetime", "timedelta"}:
        _keys(block, ["kind", "unit", "data", "timezone"] if kind == "datetime" else
              ["kind", "unit", "data"])
        if type(block["unit"]) is not str or block["unit"] not in _UNITS:
            raise ValueError("Unsupported stored timestamp unit descriptor.")
        values = _read_buffer(block["data"], "<i8", length)
        unit = block["unit"]
        if kind == "timedelta":
            return pd.Series(values.view(f"timedelta64[{unit}]"))
        timezone = block["timezone"]
        if timezone is not None and (type(timezone) is not str or not timezone):
            raise ValueError("Invalid stored timestamp timezone descriptor.")
        dates = pd.DatetimeIndex(values.view(f"datetime64[{unit}]"))
        if timezone is not None:
            try:
                dates = dates.tz_localize("UTC").tz_convert(timezone)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Invalid stored timestamp timezone descriptor.") from exc
        return pd.Series(dates)
    if kind == "category":
        _keys(block, ["kind", "ordered", "categories", "category_count", "codes"])
        if type(block["ordered"]) is not bool:
            raise ValueError("Invalid stored category ordering descriptor.")
        count = _length(block["category_count"])
        categories = _read_column(block["categories"], count, depth + 1)
        codes = _read_buffer(block["codes"], "<i8", length)
        if np.any(codes < -1) or np.any(codes >= count):
            raise ValueError("Invalid stored category codes.")
        try:
            return pd.Series(pd.Categorical.from_codes(
                codes, categories=pd.Index(categories.array, dtype=categories.dtype), ordered=block["ordered"],
            ))
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid stored category domain descriptor.") from exc
    if kind == "string":
        _keys(block, ["kind", "storage", "na_value", "values"])
        if (type(block["storage"]) is not str or block["storage"] not in {"python", "pyarrow"}
                or type(block["na_value"]) is not str or block["na_value"] not in {"NA", "nan"}):
            raise ValueError("Unsupported stored string dtype descriptor.")
        values = block["values"]
        if (type(values) is not list or len(values) != length
                or any(value is not None and type(value) is not str for value in values)):
            raise ValueError("Invalid stored string dataset values.")
        try:
            dtype = pd.StringDtype(storage=block["storage"], na_value=pd.NA if block["na_value"] == "NA" else np.nan)
            return pd.Series(values, dtype=dtype)
        except (ImportError, TypeError, ValueError) as exc:
            raise ValueError("Stored string dtype is unavailable in this environment.") from exc
    if kind == "object":
        _keys(block, ["kind", "values"])
        if type(block["values"]) is not list or len(block["values"]) != length:
            raise ValueError("Invalid stored object dataset length.")
        return pd.Series([_read_scalar(value) for value in block["values"]], dtype=object)
    raise ValueError("Unsupported stored dataset column kind.")


def pack_frame(frame):
    """Pack supported dataframe columns without float coercion or executable data."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("A dataframe is required for a dataset snapshot.")
    columns = []
    for position, name in enumerate(frame.columns):
        if type(name) is not str:
            raise ValueError("Unsupported dataset column name; column names must be strings.")
        columns.append({"name": name, **_pack_column(frame.iloc[:, position])})
    core = {"encoding": "typed-v4", "rows": len(frame), "columns": columns,
            "attrs": _attributes(frame.attrs)}
    return {**core, "sha256": _digest(core)}


def _unpack_v4(bundle):
    _keys(bundle, ["encoding", "rows", "columns", "attrs", "sha256"])
    core = {key: value for key, value in bundle.items() if key != "sha256"}
    if type(bundle["sha256"]) is not str or _digest(core) != bundle["sha256"]:
        raise ValueError("Embedded dataset integrity check failed.")
    length = _length(bundle["rows"])
    if type(bundle["columns"]) is not list:
        raise ValueError("Invalid stored dataset columns.")
    attrs = _attributes(bundle["attrs"], reading=True)
    series, names = [], []
    for block in bundle["columns"]:
        if type(block) is not dict or type(block.get("name")) is not str:
            raise ValueError("Invalid stored dataset column name descriptor.")
        names.append(block["name"])
        series.append(_read_column({key: value for key, value in block.items() if key != "name"}, length))
    frame = pd.concat(series, axis=1) if series else pd.DataFrame(index=pd.RangeIndex(length))
    frame.columns = names
    frame.attrs = attrs
    return frame


def _unpack_legacy(bundle, encoding):
    """Retain the original checksum recipes and table/overlay representations."""
    if type(bundle.get("data")) is not str or type(bundle.get("sha256")) is not str:
        raise ValueError("Invalid legacy dataset snapshot.")
    if encoding in {"typed-v2", "typed-v3"}:
        fields = ["data", "infinity", "floats"] if encoding == "typed-v3" else ["data", "infinity"]
        core = {key: bundle[key] for key in fields}
        digest = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
    else:
        digest = hashlib.sha256(bundle["data"].encode()).hexdigest()
    if digest != bundle["sha256"]:
        raise ValueError("Embedded dataset integrity check failed.")
    frame = pd.read_json(io.StringIO(bundle["data"]), orient="table")
    for pos, col, sign in bundle.get("infinity", []):
        if (type(pos) is not int or not 0 <= pos < len(frame) or col not in frame
                or type(sign) is not int or sign not in [-1, 1]):
            raise ValueError("Invalid stored infinity position.")
        frame.iat[pos, frame.columns.get_loc(col)] = np.inf * sign
    for block in bundle.get("floats", []):
        if block["column"] not in frame or block["dtype"] not in {"float64", "float32", "Float64", "Float32"}:
            raise ValueError("Invalid numeric snapshot column.")
        values = _read_buffer(block["data"], "<f8", len(frame))
        frame[block["column"]] = pd.Series(values, index=frame.index, dtype=block["dtype"])
    return frame


def unpack_frame(bundle):
    """Decode a checked snapshot; unknown encodings never enter a legacy reader."""
    try:
        if type(bundle) is not dict:
            raise ValueError("Invalid dataset snapshot.")
        encoding = bundle.get("encoding")
        if encoding == "typed-v4":
            return _unpack_v4(bundle)
        if encoding is None or (type(encoding) is str and encoding in {"typed-v1", "typed-v2", "typed-v3"}):
            return _unpack_legacy(bundle, encoding)
        raise ValueError("Unsupported dataset snapshot encoding.")
    except (KeyError, TypeError, IndexError, OverflowError, AttributeError, UnicodeError) as exc:
        raise ValueError("Invalid stored dataset snapshot descriptor.") from exc
