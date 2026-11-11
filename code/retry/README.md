# Second attempt, built-in Fourier operator

Attempt distinct from the two-layer FNO model in the parent folder. The solver uses a single interpolated transport and a small learned spectral corrector, built into the diffusion. The classical references include MacCormack and linear or cubic transport on several grids. The outcomes are on 24 new scenes.

- protocol.md: configuration and criterion fixed before the test.
- experiment.py: fit, choice on validation, benchmark and per-scene output.
- results.json: errors, latencies and comparison at similar cost.
- results/weights.json: coefficients per grid.
- results/scenes.json and results/timings.json: raw measurements.
- make_video.py: produces the visual comparison from the saved frames.

It uses the dependencies and the reference simulator of the parent folder. Set OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1. Run python experiment.py and then python make_video.py, with FFmpeg in the PATH.
