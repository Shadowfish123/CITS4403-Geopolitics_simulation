# Structural Balance and Relationship Inertia in Networks of Nations

CITS4403 Research Project. A computational model of how a network of cooperative and hostile relationships between countries settles (or fails to settle) into opposing blocs, started from real news-event data and tested under random shocks.

**All country identities are anonymised.** Countries appear only as `Nation_1 ... Nation_N` (or `C001 ...` in the stored data). No real country names appear anywhere in this repository.

## Research question
Under what conditions does a network of nations settle into at most two stable opposing blocs, and does the entrenchment of relationships (relationship inertia) change how much random disruption the network can absorb?

## What is existing work and what is ours
**Existing work (not our contribution):** structural balance theory (Heider, 1946; Cartwright and Harary, 1956) and its local-dynamics form on networks (Antal, Krapivsky and Redner, 2005): relationships in a triangle of three countries are repaired until no triangle is unbalanced. GDELT is an existing public news-event dataset.

**Our contributions:**
1. **Relationship inertia.** Each relationship carries a resistance to being flipped, derived from how much real-world coverage the pair has (log-scaled, capped at 0.9). This is an extension of the standard model.
2. **Real-data networks.** The starting network is built from GDELT snapshots (top-N countries by mention volume, signs from the relative tone of coverage, anonymised) instead of random signs.
3. **A systematic noise analysis** with confidence intervals, replicated on two snapshots of different density, with sensitivity checks on the sign threshold and control experiments (shuffled and uniform inertia, shuffled signs).
4. **A check that settling is not the same as forming two blocs** on sparse networks, using an explicit two-colouring test.

## Repository structure
```
src/
  balance_model.py      the model: signed network, triad balance, update rule, noise, inertia, simulation loop
  blocs.py              two-colouring test: is the final network a valid split into at most two blocs?
  data_pipeline.py      GDELT snapshot -> filtered, signed, anonymised network with inertia
  run_experiments.py    all experiments: noise, density, threshold, permutation test, controls
  experiments.py        the original prototype sweeps on random networks (kept for reference)
utils/
  gdelt_fetch.py        downloads and aggregates GDELT event files for one day
  anonymise_data.py     relabels countries in a raw snapshot
  check_snapshot.py     identity-free summary statistics of a snapshot
data/                   anonymised snapshots (day 1: 2026-10-04, day 2: 2026-10-06)
notebooks/analysis.ipynb  walkthrough of the data, model, experiments and results
tests/                  automated tests (pytest)
results/                tables (CSV) and figures (PNG) produced by the experiments
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
python utils/check_snapshot.py --csv data/country_relationships_day2.csv
```
**3. Quick check that the experiments run** (about 10 seconds; writes to `results_quick/`, which is git-ignored):
```bash
python src/run_experiments.py --quick --out results_quick
```
**4. Full experiments** (about 8 to 10 minutes with 4 workers; regenerates everything in `results/`):
```bash
python src/run_experiments.py --workers 4
```
Use `--only noise density threshold initial control` to run a subset and `--n 8 10` to add a 10-country network. Seeds are fixed, so results are reproducible.

**5. Notebook:** open `notebooks/analysis.ipynb` and run all cells.

## Data
The committed snapshots are anonymised copies of GDELT 2.0 event exports (one file per hour for the day, aggregated by country pair). To rebuild them from the source:
```bash
python utils/gdelt_fetch.py --date 2026-10-04 --out data/raw_gdelt_snapshot.csv
python utils/anonymise_data.py
python utils/gdelt_fetch.py --date 2026-10-06 --out data/raw_gdelt_day2.csv
python utils/anonymise_data.py --raw data/raw_gdelt_day2.csv --out data/country_relationships_day2.csv --seed 1
```
Raw files (`data/raw_*.csv`) contain real country codes and are git-ignored. Fetching needs internet access to `data.gdeltproject.org`.

## Model in brief
- **Network:** nodes are countries; each relationship is +1 (cooperative) or -1 (hostile).
- **Balance:** a triangle is balanced when the product of its three signs is +1.
- **Update rule:** each step picks a random triangle. With probability `noise` one of its relationships flips regardless of balance (a random shock). Otherwise, if the triangle is unbalanced, one relationship flips, which always restores balance.
- **Inertia:** the chosen relationship resists the flip with probability equal to its inertia.
- **Settled:** no unbalanced triangle for 800 consecutive steps. **Globally consistent:** the network can be split into at most two groups with cooperative relationships inside and hostile ones between (`blocs.py`).
- **Signs from data:** a pair is cooperative if its mean tone is above the median tone of the selected countries. Tone skews negative, so a fixed cut-off at zero would make almost every pair hostile. The 40th and 60th percentile are tested as alternatives.

## Results files
| File pattern | Content |
|---|---|
| `noise_day{1,2}_n8.*` | settling rate and time to settle against noise, with and without inertia (CSV, PNG, and every run in `*_runs.csv`) |
| `density_day{1,2}_n8.*` | settled versus globally consistent as relationships are removed |
| `threshold_day{1,2}_n8.*` | the noise experiment at the 40th, 50th and 60th percentile sign threshold |
| `control_day{1,2}_n8.*` | no inertia, real inertia, shuffled inertia, uniform inertia, shuffled signs |
| `initial_balance_day{1,2}_n8.csv` | permutation test of how balanced the real signs are |
| `tipping_points.csv` | noise level at which the settling rate falls below 50% |
| `balance_over_time.png`, `noise_sweep.png`, `density_sweep.png` | early prototype figures on random networks (from `src/experiments.py`) |

## Main findings
- Inertia raises the noise a network can absorb before it stops settling. At noise 0.015 the settling rate rises from 3% to 97% on day 1 and from 3% to 63% on day 2 (30 repeats, 95% intervals in `results/`). The effect holds on both snapshots and at all three sign thresholds.
- Without noise, every settled real-data network is a valid two-bloc split. Noise mainly decides whether the network settles, not what it settles into.
- On sparse networks, settling does not guarantee a valid split into two blocs.
- The tone-derived signs are not shown to be more balanced than chance (permutation test), and the control experiments do not show that the real-data assignment of inertia matters beyond added resistance in general.

## Limitations
Each snapshot gives one fixed 8-country starting network, so repeats vary only the random dynamics and results describe these snapshots. Tone comes from news wording, which skews negative, so signs are a media-based proxy for relationships, not diplomatic fact. Mention volume is a rough proxy for how established a relationship is. Computation time grows quickly with network size.

## Contributions
The repository was created by Shadowfish123 (initial commit and rename). Shadowfish123 also helped with the coding, writing an exploratory geography-layer module (`src/geography.py`) on the separate branch `Experimental-geography-extension`, and drafted the first version of the project report. That branch is not merged, and nothing in the reported results depends on it. All other code, tests, the data pipeline, experiments, results and documentation on `main` were committed by nikhileshvombolu.

## Tools and acknowledgements
Python, NetworkX, NumPy, Matplotlib, pytest. Data: GDELT Project (https://www.gdeltproject.org/).

## References
Antal, T., Krapivsky, P. L. and Redner, S. (2005). Dynamics of social balance on networks. *Physical Review E*, 72, 036121.
Cartwright, D. and Harary, F. (1956). Structural balance: a generalization of Heider's theory. *Psychological Review*, 63, 277-293.
Heider, F. (1946). Attitudes and cognitive organization. *Journal of Psychology*, 21, 107-112.
Leetaru, K. and Schrodt, P. A. (2013). GDELT: Global data on events, location and tone, 1979-2012. *ISA Annual Convention*.
