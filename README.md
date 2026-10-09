# Geography Extension: Relationship Reach on a Random Map

Research exploration base on main model. This extension adds the layer underneath the
structural balance model: before any balance dynamics run, geography and
economic reach decide **which pairs of nations can hold a direct relationship
at all**.

## What the extension adds

* Every nation sits at a random point on a flat map, `(x, y)` in `[0, 1]`.
* Every nation carries an economic tier: two "strong" and six "weak", assigned
  to the eight nations in a seeded random order.
* A nation reaches `r_min + beta_econ * economy` map units, so a pair is
  **admissible** when `distance(i, j) <= reach_i + reach_j`.
* Two distant weak economies can therefore never hold a direct relationship,
  while two strong economies can reach across the map. The radius `r_min` is
  calibrated by bisection so that, on average, the gate keeps the requested
  share of the real relationships. Fixing the radius by hand would give every
  snapshot and every map draw a different density, which would make the
  comparison unfair.
* A soft gate (survival probability `exp(-distance / scale)`) is kept as a
  control arm, so "far relationships are unlikely" can be compared with "far
  relationships are impossible".

## What is deliberately left unchanged

The extension is a **substrate filter**, not a change to the model:

| File | Status |
|---|---|
| `src/balance_model.py` | byte-identical to the main model |
| `src/blocs.py` | byte-identical |
| `src/data_pipeline.py` | byte-identical |
| `src/run_experiments.py` | byte-identical; its statistics are imported by the new code |
| `utils/*`, `tests/test_*.py` (original files) | byte-identical |
| `src/geography.py`, `src/geo_experiments.py`, `src/run_geo_experiments.py`, `tests/test_geography.py` | new |

`tests/test_geography.py` checks this directly: when the gate is wide enough to
keep every relationship, a model run on the gated network is *identical* to a
run on the original one, history included.



## References
Antal, T., Krapivsky, P. L. and Redner, S. (2005). Dynamics of social balance on networks. *Physical Review E*, 72, 036121.
Cartwright, D. and Harary, F. (1956). Structural balance: a generalization of Heider's theory. *Psychological Review*, 63, 277-293.
Heider, F. (1946). Attitudes and cognitive organization. *Journal of Psychology*, 21, 107-112.
Leetaru, K. and Schrodt, P. A. (2013). GDELT: Global data on events, location and tone, 1979-2012. *ISA Annual Convention*.
