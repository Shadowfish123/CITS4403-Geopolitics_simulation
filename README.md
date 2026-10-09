# Structural Balance and Relationship Inertia in Networks of Nations, with a Geography Layer

CITS4403 Research Project. A computational model of how a network of cooperative and hostile relationships between countries settles (or fails to settle) into opposing blocs, started from real news-event data, tested under random shocks, and extended with a layer that decides **which pairs of nations can hold a relationship at all**.

**All country identities are anonymised.** Countries appear only as `Nation_1 ... Nation_N` (or `C001 ...` in the stored data). No real country names appear anywhere in this repository.

## Research question
Under what conditions does a network of nations settle into at most two stable opposing blocs, and does the entrenchment of relationships (relationship inertia) change how much random disruption the network can absorb?

The geography layer adds a second question: when geography and economic reach decide which relationships are possible, does the answer to the first question change?

## What is existing work and what is ours
**Existing work (not our contribution):** structural balance theory (Heider, 1946; Cartwright and Harary, 1956) and its local-dynamics form on networks (Antal, Krapivsky and Redner, 2005): relationships in a triangle of three countries are repaired until no triangle is unbalanced. GDELT is an existing public news-event dataset. Distance decay and gravity models are standard tools in international relations; the geography layer uses the same idea of falling interaction with distance.

**Our contributions:**
1. **Relationship inertia.** Each relationship carries a resistance to being flipped, derived from how much real-world coverage the pair has (log-scaled, capped at 0.9). This is an extension of the standard model.
2. **Real-data networks.** The starting network is built from GDELT snapshots (top-N countries by mention volume, signs from the relative tone of coverage, anonymised) instead of random signs.
3. **A systematic noise analysis** with confidence intervals, replicated on two snapshots of different density, with sensitivity checks on the sign threshold and control experiments (shuffled and uniform inertia, shuffled signs).
4. **A check that settling is not the same as forming two blocs** on sparse networks, using an explicit two-colouring test.
5. **A geography layer.** Every nation sits at a random point on a flat map and carries an economic tier; a pair can hold a direct relationship only when the distance between them is within the sum of the two reaches, `d_ij <= R_i + R_j` with `R_i = r_min + beta * economy_i`. The radius is calibrated so the gate keeps a set share of the real relationships. The layer only removes relationships: signs and inertia are copied unchanged, no relationship is added, and the model, the bloc test and the data pipeline are byte-identical to the version without it.

## Repository structure
```
src/
  balance_model.py      the model: signed network, triad balance, update rule, noise, inertia, simulation loop
  blocs.py              two-colouring test: is the final network a valid split into at most two blocs?
  data_pipeline.py      GDELT snapshot -> filtered, signed, anonymised network with inertia
  run_experiments.py    the original experiments: noise, density, threshold, permutation test, controls
  geography.py          the geography layer: positions, economic tiers, reach, hard and soft gate, radius calibration
  geo_experiments.py    experiments on gated networks: substrates, sweeps, control arms, figures
  run_geo_experiments.py  driver for the geography experiments
  experiments.py        the original prototype sweeps on random networks (kept for reference)
utils/
  gdelt_fetch.py        downloads and aggregates GDELT event files for one day
  anonymise_data.py     relabels countries in a raw snapshot
  check_snapshot.py     identity-free summary statistics of a snapshot
data/                   anonymised snapshots (day 1: 2026-10-04, day 2: 2026-10-06)
notebooks/analysis.ipynb  walkthrough of the data, the original model, the original results, and the geography comparison
tests/                  automated tests, including test_geography.py
results/                tables (CSV) and figures (PNG) of the original experiments
results_geo/            tables (CSV) and figures (PNG) of the geography experiments
README_GEO.md           the geography layer in detail: rules, design choices, full result tables
requirements.txt
```

## Setup
Requires Python 3.8 or later (developed and tested on 3.10 to 3.12).
```bash
pip install -r requirements.txt
```
On Windows, use `py -m pip install -r requirements.txt` and `py` in place of `python` below. To open the notebook you also need Jupyter (`pip install jupyter`).

## Reproducing the results
**1. Run the tests** (about 2 seconds):
```bash
python -m pytest tests -q
```
**2. Summarise a data snapshot** (prints counts only, no identities):
```bash
python utils/check_snapshot.py
```
**3. The original experiments** (about 8 to 10 minutes with 4 workers; writes to `results/`):
```bash
python src/run_experiments.py --quick --out results_quick
python src/run_experiments.py --workers 4
```
**4. The geography experiments** (writes to `results_geo/`):
```bash
python src/run_geo_experiments.py --quick
python src/run_geo_experiments.py --draws 30 --repeats 5 --workers 4
python src/run_geo_experiments.py --only keep --draws 20 --repeats 4
```
The map is a random draw, so every arm is repeated over many position draws and the tables report the spread across draws as well as the pooled rate. Use `--only noise gate space economy keep` to run a subset.

**5. Notebook:** open `notebooks/analysis.ipynb` and run all cells. Sections 1 to 5 present the original model and its results; sections 6 to 9 add the geography layer and compare the two; sections 10 and 11 discuss limitations and reproduce everything.

## Data
The committed snapshots are anonymised copies of GDELT 2.0 event exports (one file per hour for the day, aggregated by country pair). To rebuild them from the source:
```bash
python utils/gdelt_fetch.py --date 2026-10-04 --out data/raw_gdelt_snapshot.csv
python utils/anonymise_data.py
python utils/gdelt_fetch.py --date 2026-10-06 --out data/raw_gdelt_day2.csv
python utils/anonymise_data.py --raw data/raw_gdelt_day2.csv --out data/country_relationships_day2.csv --seed 1
```
Raw files (`data/raw_*.csv`) contain real country codes and are git-ignored. Fetching needs internet access to `data.gdeltproject.org`. Note that the anonymisation step does not keep the code-to-label mapping, so any additional country-level data has to be joined before anonymisation.

## Model in brief
- **Network:** nodes are countries; each relationship is +1 (cooperative) or -1 (hostile).
- **Balance:** a triangle is balanced when the product of its three signs is +1.
- **Update rule:** each step picks a random triangle. With probability `noise` one of its relationships flips regardless of balance (a random shock). Otherwise, if the triangle is unbalanced, one relationship flips, which always restores balance.
- **Inertia:** the chosen relationship resists the flip with probability equal to its inertia.
- **Settled:** no unbalanced triangle for 800 consecutive steps. **Globally consistent:** the network can be split into at most two groups with cooperative relationships inside and hostile ones between (`blocs.py`).
- **Signs from data:** a pair is cooperative if its mean tone is above the median tone of the selected countries. Tone skews negative, so a fixed cut-off at zero would make almost every pair hostile.

**Geography layer.** Each nation sits at a random point on a flat map and carries an economic tier (two strong, six weak, assigned at random). A nation reaches `R_i = r_min + beta * economy_i` map units, and a pair is admissible when `d_ij <= R_i + R_j`, so two distant weak economies can never hold a direct relationship while two strong economies can reach across the map. `r_min` is calibrated by bisection so that the gate keeps a set share of the real relationships (75% by default); fixing it by hand would give every map draw a different density. A soft gate, where a relationship survives with probability `exp(-d/scale)`, is kept as a control arm.




## References
Antal, T., Krapivsky, P. L. and Redner, S. (2005). Dynamics of social balance on networks. *Physical Review E*, 72, 036121.
Cartwright, D. and Harary, F. (1956). Structural balance: a generalization of Heider's theory. *Psychological Review*, 63, 277-293.
Heider, F. (1946). Attitudes and cognitive organization. *Journal of Psychology*, 21, 107-112.
Leetaru, K. and Schrodt, P. A. (2013). GDELT: Global data on events, location and tone, 1979-2012. *ISA Annual Convention*.
