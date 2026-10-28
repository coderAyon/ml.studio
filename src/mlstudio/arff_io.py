"""Dense Weka ARFF exchange without evaluation or executable parser extensions.

Numeric, nominal and string attributes are supported. Sparse rows, instance weights,
DATE and RELATIONAL declarations are rejected explicitly so unsupported data cannot
silently become a different dataset. Missing values are unquoted question marks;
quoted question marks are ordinary strings.
"""
from dataclasses import dataclass
from datetime import date, datetime
import math
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd


MAX_FILE_BYTES = 250_000_000


@dataclass(frozen=True)
class _Token:
    value: str
    quoted: bool = False


def _commentless(line):
    quote = None
    escaped = False
    for index, character in enumerate(line):
        if escaped:
            escaped = False
        elif character == "\\" and quote:
            escaped = True
        elif quote:
            if character == quote:
                quote = None
        elif character in "\"'":
            quote = character
        elif character == "%":
            return line[:index].strip()
    return line.strip()


def _consume(text, comma=False):
    text = text.lstrip()
    if not text:
        raise ValueError("An ARFF name or value is empty; use ? for missing data.")
    if text[0] not in "\"'":
        stop = next((i for i, c in enumerate(text) if c == "," or (not comma and c.isspace())), len(text))
        value = text[:stop].strip()
        if not value:
            raise ValueError("An ARFF value is empty; use ? for missing data.")
        return _Token(value), text[stop:]
    quote = text[0]
    output = []
    index = 1
    escapes = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f", "\\": "\\", "'": "'", '"': '"', "%": "%"}
    while index < len(text):
        character = text[index]
        if character == quote:
            return _Token("".join(output), True), text[index + 1:]
        if character == "\\":
            index += 1
            if index == len(text) or text[index] not in escapes:
                raise ValueError("Unsupported ARFF quoted escape. Escape backslashes as \\\\ and use \\n for a line break.")
            character = escapes[text[index]]
        output.append(character)
        index += 1
    raise ValueError("An ARFF quoted value is not closed. Use escaped line breaks inside quoted strings.")


def _fields(text):
    values = []
    while True:
        token, rest = _consume(text, comma=True)
        values.append(token)
        rest = rest.strip()
        if not rest:
            return values
        if not rest.startswith(","):
            raise ValueError("Separate ARFF values with commas after a quoted value.")
        text = rest[1:]


def load_arff(path, encoding="utf-8-sig"):
    """Read a dense ARFF file, retaining declared nominal values and column order."""
    source = Path(path)
    if source.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("ARFF files support up to 250 MB. Use a smaller extract.")
    relation = None
    attributes = []
    rows = []
    data_started = False
    with source.open("r", encoding=encoding, newline="") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = _commentless(raw)
            if not line:
                continue
            try:
                if not data_started:
                    match = re.match(r"@(\w+)\b(.*)$", line)
                    if not match:
                        raise ValueError("Expected @RELATION, @ATTRIBUTE or @DATA in the ARFF header.")
                    directive, body = match[1].lower(), match[2].strip()
                    if directive == "relation":
                        token, rest = _consume(body)
                        if relation is not None or rest.strip():
                            raise ValueError("Declare one ARFF relation name; quote names containing spaces.")
                        relation = token.value
                    elif directive == "attribute":
                        token, declaration = _consume(body)
                        if not token.value or token.value in [a["name"] for a in attributes]:
                            raise ValueError("ARFF attribute names must be nonempty and unique.")
                        declaration = declaration.strip()
                        kind = declaration.lower()
                        attribute = {"name": token.value}
                        if kind in {"numeric", "real", "integer"}:
                            attribute["kind"] = "numeric"
                        elif kind == "string":
                            attribute["kind"] = "string"
                        elif declaration.startswith("{") and declaration.endswith("}"):
                            values = [item.value for item in _fields(declaration[1:-1])]
                            if len(values) != len(set(values)):
                                raise ValueError("Nominal ARFF values must be distinct.")
                            attribute.update(kind="nominal", values=values)
                        elif kind.startswith("date"):
                            raise ValueError("ARFF DATE attributes are not supported yet. Export dates as ISO STRING values or CSV first.")
                        elif kind.startswith("relational"):
                            raise ValueError("Relational ARFF attributes are not supported. Flatten nested records into a table first.")
                        else:
                            raise ValueError("Use ARFF numeric, nominal {value,...}, or string attribute types.")
                        attributes.append(attribute)
                    elif directive == "data":
                        if body or relation is None or not attributes:
                            raise ValueError("ARFF needs @RELATION and at least one @ATTRIBUTE before an empty @DATA declaration.")
                        data_started = True
                    else:
                        raise ValueError(f"Unsupported ARFF header directive @{directive}.")
                else:
                    if line.startswith("{"):
                        raise ValueError("Sparse ARFF rows are not supported yet. Export dense ARFF or CSV first.")
                    tokens = _fields(line)
                    if len(tokens) != len(attributes):
                        extra = " Instance weights are not supported." if tokens[-1].value.startswith("{") else ""
                        raise ValueError(f"Expected {len(attributes)} ARFF values; found {len(tokens)}.{extra}")
                    row = []
                    for attribute, token in zip(attributes, tokens):
                        if token.value == "?" and not token.quoted:
                            row.append(np.nan)
                        elif attribute["kind"] == "numeric":
                            try:
                                value = float(token.value)
                            except ValueError as exc:
                                raise ValueError(f"Attribute {attribute['name']!r} needs a number or ?.") from exc
                            if not math.isfinite(value):
                                raise ValueError(f"Attribute {attribute['name']!r} needs a finite number; use ? for missing data.")
                            row.append(value)
                        elif attribute["kind"] == "nominal" and token.value not in attribute["values"]:
                            raise ValueError(f"Value {token.value!r} is outside the declared domain of {attribute['name']!r}.")
                        else:
                            row.append(token.value)
                    rows.append(row)
            except ValueError as exc:
                raise ValueError(f"ARFF line {line_number}: {exc}") from exc
    if not data_started:
        raise ValueError("ARFF is missing its @DATA section.")
    if not rows:
        raise ValueError("The ARFF file has no data rows.")
    frame = pd.DataFrame(rows, columns=[a["name"] for a in attributes])
    for attribute in attributes:
        if attribute["kind"] == "nominal":
            frame[attribute["name"]] = pd.Categorical(frame[attribute["name"]], categories=attribute["values"])
    frame.attrs.update(arff_relation=relation, arff_schema=attributes, duplicate_headers=[])
    return frame


def _quote(value):
    text = str(value)
    for before, after in [("\\", "\\\\"), ("'", "\\'"), ("\n", "\\n"), ("\r", "\\r"), ("\t", "\\t"), ("\b", "\\b"), ("\f", "\\f")]:
        text = text.replace(before, after)
    return "'" + text + "'"


def _scalar_text(value):
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (str, bool, int, float, np.integer, np.floating, np.bool_)):
        return str(value)
    raise ValueError("ARFF export supports scalar numbers, text, categories and dates. Convert nested/object values to text first.")


def export_arff(frame, path, relation="MLStudio"):
    """Write a dense ARFF atomically. Datetime columns use ISO STRING values."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Keep at least one row and column before ARFF export.")
    names = list(map(str, frame.columns))
    if len(names) != len(set(names)) or any(not name for name in names):
        raise ValueError("ARFF export needs nonempty, unique column names.")
    declarations = []
    kinds = []
    for column, name in zip(frame.columns, names):
        series = frame[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            domain = [_scalar_text(value) for value in series.cat.categories]
            if len(domain) != len(set(domain)):
                raise ValueError(f"Column {name!r} has categories that become identical text. Rename them before export.")
            declarations.append("{" + ",".join(_quote(value) for value in domain) + "}" if domain else "STRING")
            kinds.append("string")
        elif pd.api.types.is_bool_dtype(series):
            declarations.append("{'False','True'}")
            kinds.append("string")
        elif pd.api.types.is_numeric_dtype(series):
            declarations.append("NUMERIC")
            kinds.append("numeric")
        else:
            declarations.append("STRING")
            kinds.append("string")
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=destination.parent,
                                         prefix=".mlstudio-arff-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write("@RELATION " + _quote(relation) + "\n\n")
            for name, declaration in zip(names, declarations):
                output.write("@ATTRIBUTE " + _quote(name) + " " + declaration + "\n")
            output.write("\n@DATA\n")
            for row in frame.itertuples(index=False, name=None):
                cells = []
                for name, kind, value in zip(names, kinds, row):
                    missing = pd.isna(value)
                    if not isinstance(missing, (bool, np.bool_)):
                        raise ValueError(f"Column {name!r} contains a nested value. Convert it to text before export.")
                    if missing:
                        cells.append("?")
                    elif kind == "numeric":
                        if not np.isfinite(value):
                            raise ValueError(f"Column {name!r} contains infinity. Replace it with a finite value or missing before export.")
                        cells.append(str(value))
                    else:
                        cells.append(_quote(_scalar_text(value)))
                output.write(",".join(cells) + "\n")
        temporary.replace(destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
