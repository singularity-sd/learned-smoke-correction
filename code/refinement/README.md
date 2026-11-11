# Spectral operator with a midpoint trajectory

Study distinct from the two earlier pilots. It combines a temporal estimate of the velocity field, a midpoint backtrace and a learned Fourier filter. The filter is built into the diffusion and uses eight trained coefficients. The trajectory and the filter act within every step, without access to future frames.

- development.md and development.py: training and development comparison on the earlier seeds.
- results/weights.json and results/development.json: weights, error and latency on validation.
- protocol.md and confirmation.py: new test on 64 independent scenes.
- results.json, results/scenes.json and results/timing_seconds.json: aggregate, per-scene and per-frame outcomes.
- make_video.py: side-by-side comparison on the first scenes of the two regimes.

Dependencies as in the parent folder. It reuses the simulator and the earlier filter from ../retry. Set OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1. Run python development.py, then python confirmation.py and python make_video.py. The video needs FFmpeg.

## Finer reference

The protocol and the results of the 192 x 192 test are in protocol_fine.md, confirmation_fine.py and results_fine.json. The checkpoint stays the same. The new implementation factorises the filter by axis. The 256 x 256 check and the per-scene outcomes are in results/fine. Run python confirmation_fine.py and python make_video_fine.py in the same environment.
