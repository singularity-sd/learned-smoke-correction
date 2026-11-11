# Neural operators in computer graphics

Pilot of the forecast T3-2026-0032. A Fourier neural operator corrects the physical step on a 48x48 grid at every frame. The model changes the dynamics of density and vorticity. All sequences are generated with the code included here.

- protocol.md: per-scene split, comparisons and threshold fixed before the test.
- experiment.py: numerical simulator, data generation, training and benchmark.
- results.json and results/scenes.json: aggregate and per-scene metrics.
- results/timing_seconds.json: CPU times per single frame.
- results/operator.pt: checkpoint selected on the validation rollout.
- make_demo.py: exports a side-by-side video of the first scenes of the two test groups.
- data/train.npz: regenerable cache of the training steps.

Python 3.12, NumPy 2.5.4, SciPy 1.18.1, PyTorch 2.14.1 CPU, OpenCV headless 4.13.0.92. Create data and results, then run python experiment.py. The command python make_demo.py needs FFmpeg in the PATH. The run uses four PyTorch threads and one OpenCV thread. The benchmark covers synthetic 2D smoke in a periodic domain without obstacles.

The two later designs are in `retry/` and `refinement/`; the study reported in the paper is the one in `refinement/`.
The files `results/before_test.csv` (here and in the subfolders) list the SHA-256 fingerprints of the protocol and code
taken before each test. They refer to the original files, which had Italian names and comments; the copies here were
later translated into English (names, comments and messages only), so their fingerprints differ.
