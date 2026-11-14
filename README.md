# Goal-derived information constraints for dual control

Code and LaTeX sources for the L4DC 2027 submission *Goal-derived information constraints for dual control* (W. M. Kouw, Eindhoven University of Technology).

The paper treats dual control as constrained Bethe free energy minimisation (CBFE) on the factor graph of a planning model. An information constraint between the next output and the beliefs over the unknown parameters and states takes its floor from the goal prior. Its KKT multiplier acts as a probing weight. The weight is positive while parameter uncertainty spreads the predictions beyond the goal tolerance, and it vanishes once it does not. The controller therefore needs no tuned probing weight.

- Paper: [`main.pdf`](main.pdf) (source [`main.tex`](main.tex))
- Supplementary material: [`supplementary.pdf`](supplementary.pdf) (source [`supplementary.tex`](supplementary.tex))

## Where to find each result

| Paper | What it shows | Code |
|---|---|---|
| Section 5, Fig. 2 | Quadcopter with an unknown compass fault and wind | [`experiment-ablation/`](experiment-ablation), agent in [`agents/cbfe_quad/`](agents/cbfe_quad) |
| Supplement S2.1–S2.4, Table S1 | Simulator, planning model, message passing and baselines of the quadcopter | [`agents/cbfe_quad/agent.py`](agents/cbfe_quad/agent.py), [`agents/cbfe_quad/messages.py`](agents/cbfe_quad/messages.py), [`experiment-ablation/ablation.py`](experiment-ablation/ablation.py) |
| Supplement S2.3, last paragraph | Message passing planner against L-BFGS-B | [`experiment-ablation/run_message.py`](experiment-ablation/run_message.py) |
| Supplement S2.5 | Protocol, seed sets and the logged setting search | [`experiment-ablation/search.py`](experiment-ablation/search.py), [`experiment-ablation/paper_data.py`](experiment-ablation/paper_data.py) |
| Supplement S3.1, Fig. S1 | Integrator with unknown gain (Åström–Helmersson benchmark) | [`experiment-integrator/`](experiment-integrator), agent in [`agents/cbfe_arx/`](agents/cbfe_arx) |
| Supplement S3.2, Fig. S2 | Light–dark positioning with a hidden state | [`experiment-lightdark/`](experiment-lightdark), agent in [`agents/cbfe_lightdark/`](agents/cbfe_lightdark) |

Every script starts with a docstring that states what it computes, which files it reads and writes, and which part of the paper it belongs to.

### Names in the code and in the paper

The quadcopter code predates the final names of the controllers.

| Paper | Label in `experiment-ablation/` |
|---|---|
| info-CBFE | `CBFE` |
| no-info-CBFE | `no constraint` |
| dual MPC | `dual MPC` |
| MPC-X | `MPC-X` |
| oracle (knows wind and fault) | `known model` |

The runs also contain two arms that the paper does not report: `no trace` (constraint without the caution term) and `certainty equivalence`.

## Installation

The experiments were run with Python 3.14 on Linux. They need

```
numpy>=2.4  scipy>=1.18  casadi>=3.8  pandas>=3.0  matplotlib>=3.11
```

for example `pip install numpy scipy casadi pandas matplotlib`. Run every command below from the repository root. The scripts put the repository root on `sys.path` themselves, so `agents` imports without installation.

## Reproducing the results

Simulated flights of the quadcopter and the integrator are not in the repository (`data/` and `results/` of these experiments are git-ignored), so their figures require rerunning the experiments. The light–dark results are included.

### Quadcopter (Section 5, Fig. 2)

```bash
python experiment-ablation/run.py --fresh                  # all CBFE variants and the oracle -> data/episodes_fresh.pkl
python experiment-ablation/run_baselines.py transfer       # tune rho and kappa on held-out seeds 3000-3049
python experiment-ablation/run_baselines.py test --fresh   # tuned dual MPC and MPC-X -> data/baselines_test_fresh.pkl
python experiment-ablation/plot_over_time.py --fresh       # Fig. 2(a)-(c) and the legend
python experiment-ablation/plot.py --fresh                 # Fig. 2(d)
python experiment-ablation/summary.py --fresh              # numbers of the Results paragraph -> results/summary_fresh.csv
```

The quadcopter uses these seed sets (supplement S2.5):

- 2000–2029: the logged search for the setting (`search.py`);
- 3000–3049: tuning of the dual MPC weight ρ and the MPC-X accuracy κ;
- 3000–3099: validation of the setting;
- 1000–1099: test seeds, which also informed the choice of the setting;
- 4000–4099: fresh seeds that played no role in the design. These are the seeds the paper reports.

`paper_data.py` drops seed 4076, on which even the oracle ends farther from the target than it started. Without `--fresh`, the same scripts run on the test seeds 1000–1099.

Wall-clock times in Fig. 2(c)–(d) come from an AMD Ryzen AI 9 HX 370. At the nominal IMU noise, the scripts pin the flights to cores 0–3 (its four Zen 5 cores) with `os.sched_setaffinity`, so that the times compare between controllers. The other noise levels run on twelve processes. `run.py` takes about half an hour on this machine. Times on other machines will differ in scale, but they should keep the same ordering.

The message passing validation of supplement S2.3:

```bash
python experiment-ablation/run_message.py shadow   # both planners on identical beliefs, seeds 1000-1009
python experiment-ablation/run_message.py test     # closed-loop message passing planner, seeds 1000-1099
python experiment-ablation/run_message.py report
```

### Integrator (supplement S3.1, Fig. S1)

```bash
python experiment-integrator/run.py tune     # tune rho and the MPC-X accuracy on held-out seeds
python experiment-integrator/run.py          # all controllers -> results/benchmark.pkl
python experiment-integrator/plot_paper.py   # fig_integrator_sensitivity.pdf, fig_integrator_constraint.pdf
python experiment-integrator/analysis.py     # numbers of the section
```

The first run solves the dynamic program of the optimal dual controller (about three minutes on 24 cores) and caches it.

### Light–dark (supplement S3.2, Fig. S2)

The stored results allow the figure to be redrawn directly:

```bash
python experiment-lightdark/performance.py      # G per episode -> results/baselines_performance.pkl
python experiment-lightdark/plot_baselines.py   # fig_lightdark_{legend,time,noise,wallclock}.pdf
```

To rerun the episodes, call `python experiment-lightdark/run_baselines.py tune` and then `python experiment-lightdark/run_baselines.py test` before the two commands above.

## Repository layout

```
main.tex, supplementary.tex   paper and supplement (PMLR / L4DC format, l4dc2027.cls)
references.bib                bibliography of both
graphs/                       TikZ sources of the factor graph (Fig. 1)
figures/paper_style.py        shared matplotlib style, colours and legends of all figures
agents/                       controllers
  cbfe_quad/                    CBFE, dual MPC and MPC-X for the quadcopter, with the extended Kalman filter
  cbfe_arx/, dual_mpc/, mpcx/   CBFE and baselines on a Bayesian ARX model (integrator)
  cbfe_lightdark/               CBFE on a grid filter for the light-dark task
  common.py                     code shared by the agents
experiment-ablation/          quadcopter experiment (main paper)
experiment-integrator/        integrator experiment (supplement)
experiment-lightdark/         light-dark experiment (supplement)
```

Each experiment directory has its own `figures/`, `results/` and, where needed, `data/`. LaTeX finds the figures through `\graphicspath`.

The following parts of the repository are not used in the submission. They are kept for transparency about the development history:

- `experiment-arm/` and `experiment-planar/`: earlier candidate experiments (a two-link arm with unknown motor polarity, and planar positioning);
- `agents/cbfe_fault/`, `agents/cbfe_nlssm/`, `agents/cbfe_ssm/`, `agents/pilco/` and `agents/tdmpc2/`: agents of these and earlier experiments;
- `figures/candidates/` directories and the `pilot_*.py` scripts: alternative panels and pilot runs;
- quadcopter scripts `free_energy.py`, `plot_total.py`, `plot_by_fault.py` and `plot_free_energy.py`, which draw alternative views of the same flights.

## Building the paper

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supplementary && bibtex supplementary && pdflatex supplementary && pdflatex supplementary
```

Build `main.tex` first. The supplement reads equation, theorem and section numbers from `main.aux` via the `xr` package.

## License

MIT, see [`LICENSE`](LICENSE).
