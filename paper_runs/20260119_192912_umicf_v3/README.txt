Paper-style baseline run (UMi cell-free uplink)
=================================================

Output folder: paper_runs/20260119_192912_umicf_v3

Common env config:
- Scenario: 3GPP TR 38.901 UMi (uplink), street deployment
- A=32 APs, U=16 UEs, K=16 RB-slices
- area=200m x 200m, street_block=50m, indoor_prob=0.2
- AP height=10m, UE height=1.5m
- Nr=16 (AP rx ants), Nt=1 (UE tx ants)
- RB bandwidth W=180 kHz, NF=7 dB, pmax=0.2 W
- Carrier frequencies: 3.5GHz, 7GHz, 28GHz
- steps(CDF)=300, steps(PF curve)=400, batch_size=1
- Focus settings: [(4, 1, 'pf', True), (4, 1, 'gain', False), (4, 1, 'optimal', False), (8, 1, 'pf', True), (8, 2, 'pf', True)]
- Reps for CDF plots: [(4, 1, 'pf', 1), (4, 1, 'optimal', 0), (4, 1, 'gain', 0)]

Files:
- paper_results.csv : summary table (mean/p50/p95/fairness/nan_count)
- data/raw_samples.npz : sum-rate + per-UE raw samples for all focus settings
- data/pf_curves.npz : PF avg_rate curves (EMA state) for each fc
- figs/*.png : all plots

Notes:
- CDF plots flatten per-UE throughput across (steps x U) samples (paper common).
- PF convergence plot uses mean(avg_rate) vs step (clean paper style).
