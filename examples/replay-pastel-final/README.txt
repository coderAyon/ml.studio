ML Studio Research Pack
Contains the active dataset, transformed values, positional split definitions and original-source provenance. Treat it as private data. No model pickle is included.

Install the exact requirements into a compatible Python environment, then run: python reproduce.py
The replay checks both dataset fingerprints and split definitions, reruns the recorded experiments and compares metrics with tolerance 1e-9. Runtime timing is not expected to match. Final-test errors are descriptive; model choices informed by them require new external data.

Sources are included; this is a generated Python project. Inspect scripts before executing research packs obtained from other people.
