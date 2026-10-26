"""CSV loading and immutable-by-convention dataset versions."""
from dataclasses import dataclass
from pathlib import Path
import csv
import pandas as pd


def load_csv(path: str, delimiter: str | None = None, encoding: str = "utf-8-sig", header: bool = True, decimal: str = ".") -> pd.DataFrame:
    source = Path(path)
    if source.stat().st_size > 250_000_000:
        raise ValueError("This release supports CSV files up to 250 MB. Use a smaller extract.")
    with source.open("r", encoding=encoding, newline="") as handle:
        sample = handle.read(65536)
    if not sample.strip():
        raise ValueError("The CSV file is empty.")
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    frame = pd.read_csv(source, sep=delimiter, encoding=encoding, header=0 if header else None, decimal=decimal, on_bad_lines="error")
    if frame.empty:
        raise ValueError("The CSV has no data rows.")
    frame.columns = frame.columns.map(str)
    if header:
        raw_names = next(csv.reader(sample.splitlines(), delimiter=delimiter))
        frame.attrs["duplicate_headers"] = sorted({name for name in raw_names if raw_names.count(name) > 1})
    return frame


@dataclass
class Transformation:
    operation: str
    columns: list[str]
    value: str = ""

    def preview(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.operation in {"Cap outliers (IQR)","Remove outlier rows (IQR)","Drop missing rows"}:
            from .preprocessing import apply_operation
            return apply_operation(frame,self.operation,self.columns[0] if self.columns else '',self.value)
        result = frame.copy(deep=True)
        if self.operation == "Remove duplicates":
            return result.drop_duplicates().reset_index(drop=True)
        if self.operation == "Drop columns":
            return result.drop(columns=self.columns)
        if self.operation == "Filter rows":
            column = self.columns[0]
            return result[result[column].astype("string").str.contains(self.value, regex=False, case=False, na=False)].reset_index(drop=True)
        if self.operation == "Rename column":
            name = self.value.strip()
            if not name or name in result.columns:
                raise ValueError("Choose a nonempty, unique column name.")
            return result.rename(columns={self.columns[0]: name})
        for column in self.columns:
            series = result[column]
            if self.operation == "Fill missing":
                value = series.median() if pd.api.types.is_numeric_dtype(series) else (series.mode().iloc[0] if not series.mode().empty else "Missing")
                result[column] = series.fillna(value)
            elif self.operation == "Trim whitespace":
                result[column] = series.map(lambda x: x.strip() if isinstance(x, str) else x)
            elif self.operation == "Lowercase labels":
                result[column] = series.map(lambda x: x.lower() if isinstance(x, str) else x)
            elif self.operation == "Convert to numeric":
                result[column] = pd.to_numeric(series, errors="raise")
            elif self.operation == "Convert to text":
                result[column] = series.astype("string")
            elif self.operation == "Convert to datetime":
                result[column] = pd.to_datetime(series, errors="raise", format="mixed")
            elif self.operation == "Replace infinity with missing":
                result[column] = series.replace([float("inf"),float("-inf")],float("nan"))
            else:
                raise ValueError(f"Unsupported transformation: {self.operation}")
        return result


class DatasetVersions:
    def __init__(self, frame: pd.DataFrame):
        self.versions = [frame.copy(deep=True)]
        self.index = 0
        self.log: list[dict] = []

    @property
    def current(self):
        return self.versions[self.index].copy(deep=True)

    def apply(self, operation: Transformation):
        new = operation.preview(self.current)
        if new.empty:
            raise ValueError("This change would leave no data. Keep at least one row and column.")
        self.versions = self.versions[:self.index + 1] + [new]
        self.log = self.log[:self.index] + [vars(operation)]
        self.index += 1

    def undo(self):
        self.index = max(0, self.index - 1)

    def redo(self):
        self.index = min(len(self.versions) - 1, self.index + 1)
