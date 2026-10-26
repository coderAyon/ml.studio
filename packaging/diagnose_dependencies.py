"""Inspect bundled Qt's static dependency symbols after a Windows build."""
from pathlib import Path
import pefile

root = Path(__file__).resolve().parents[1] / "dist/MLStudio-Modern/_internal"
files = {p.name.lower(): p for p in root.rglob("*") if p.is_file()}
cache = {}
for name in ["qtcore.pyd", "qt6core.dll", "pyside6.abi3.dll", "shiboken6.abi3.dll"]:
    pe = pefile.PE(str(files[name]), max_symbol_exports=100000)
    for dependency in pe.DIRECTORY_ENTRY_IMPORT:
        dll = dependency.dll.decode().lower()
        if dll not in files:
            continue
        if dll not in cache:
            module = pefile.PE(str(files[dll]), max_symbol_exports=100000)
            cache[dll] = {symbol.name for symbol in module.DIRECTORY_ENTRY_EXPORT.symbols}
        missing = [symbol.name.decode() for symbol in dependency.imports if symbol.name and symbol.name not in cache[dll]]
        if missing:
            print(name, "requires missing exports in", files[dll], missing, flush=True)
