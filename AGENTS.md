This repo implements FBU (Fairness Bonded Utility), Wang et al., FAccT 2023.

- docs/FBU_IMPLEMENTATION.md is authoritative. It resolves ambiguities in the
  paper. Do NOT re-derive definitions from the PDF.
- The fairness axis is ALWAYS higher-is-better: fairness_score = 1 - abs(metric).
  Any code putting a raw signed SPD/EOD on that axis is a bug.
- Regions 1/3/5 are relative to the ORIGINAL MODEL point.
  Regions 2/4 are relative to the BASELINE CURVE.
  Do not collapse these into one reference.
- Never use accuracy, F1, or precision as a performance metric. See spec 2.2.
- No new dependencies without asking.
  numpy / pandas / scikit-learn / matplotlib / pytest / hypothesis only.
- Any modelling choice not stated in the paper carries a `# [D#]` comment
  referencing docs/DEVIATIONS.md.
- Tests first. Golden values must be hand-computable and derived in the docstring.
- Type hints on all public APIs. No bare `Any`.
