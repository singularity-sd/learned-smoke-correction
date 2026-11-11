# A compact learned Fourier correction for two-dimensional smoke simulation

From a v1 forecast: 8.8% less error than the best classical solver at the same speed, with eight learned numbers.

**Paper:** [paper.pdf](paper.pdf) · **Page:** [singularitysd.com/research/smoke.html](https://singularitysd.com/research/smoke.html) · Preprint, not yet peer reviewed.

## Abstract

A neural operator placed after a complete smoke solver can cost more than simply using a finer classical grid. We instead fold a learned Fourier correction with eight coefficients into the diffusion transform of a one-step transport solver with a midpoint trajectory. On 64 held-out synthetic smoke scenes the method ran at 0.889 ms per frame on one CPU and reduced mean density error by 8.8% relative to the best classical configuration within the same latency band (0.955 ms), with a paired interval of 4.6 to 13.1%; it improved 49 of 64 scenes and ran 55 times faster than the fine reference. The criterion was fixed before the test. The gain grows with the accuracy of the reference (9.8% against a $256\times256$ reference on eight predetermined scenes, 3.0% against a coarser $144\times144$ one), consistent with the correction recovering detail that coarse grids lose. The learned part adds only eight numbers to an existing solver.

## How it was made

This study started from the Singularity forecast **T3-2026-0032**: a method predicted to enter a new field. It was then tested once, with the plan fixed before looking at the results.

## What is in this repository

| file | what it is |
|---|---|
| `code/` | full code, protocol and intermediate results |
| `experiment.py` | reproduces the study end to end |
| `fig_frame.png` | figure |
| `main.tex` | LaTeX source of the paper |
| `paper.pdf` | the paper |
| `results.json` | every number reported in the paper |

## Reproduce

```bash
pip install numpy scipy torch
python experiment.py
```

It writes `results.json` and the figures next to the script.

## Cite

```bibtex
@misc{singularity_learned_smoke_correction_2026,
  title  = {A compact learned Fourier correction for two-dimensional smoke simulation},
  author = {Singularity},
  year   = {2026},
  url    = {https://github.com/singularity-sd/learned-smoke-correction}
}
```

## License

Code: MIT (`LICENSE`). Paper, text and figures: CC BY 4.0 (`LICENSE-paper.md`). Third-party data keep their own licence where stated.

---

Part of [Singularity](https://singularitysd.com): forecasting where science goes next, then testing it.
