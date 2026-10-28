#!/bin/bash
set -e
echo "Starting plots:"
echo "[1/4] Plotting Dynamics..."
python3 -m val.closed_loop.plot_dynamics
echo "[2/4] Plotting Single Attractor..."
python3 -m val.closed_loop.plot_attractor
echo "[3/4] Plotting Sampling Attractors..."
python3 -m val.closed_loop.sample_attractors
echo "[4/4] Plotting Invariant Measure..."
python3 -m val.closed_loop.plot_invariant_measure
echo "Plotting successful!"
