# NGT Noise Suppression Research

Real-time speech enhancement for G.711 A-law telephony VoIP systems,
deployed on Raspberry Pi CM5 via NetGeneTech RAU-VCU pipeline.

## Paper Target
MDPI Electronics / Sensors or Computer Standards & Interfaces

## Novel Contributions
1. Bounded Causal State Unit (BCSU) — frame-causal architecture derived
   from 20ms RTP timing constraint
2. A-law Domain Loss — codec-weighted loss function for G.711 environments
3. Real-time deployment on CM5 embedded hardware (validated)

## Project Structure
- `src/data/`     — download, A-law preprocessing, splitting
- `src/models/`   — all 5 model architectures
- `src/training/` — training loop, loss functions, config
- `src/utils/`    — shared audio and codec utilities
- `notebooks/`    — feasibility check and final evaluation
- `outputs/stats/`— PESQ/STOI results CSVs (git tracked)
- `deploy/`       — CM5 real-time inference integration

## Setup
```bash
git clone https://github.com/chinimini532/ngt-ns-research.git
cd ngt-ns-research
pip install -r requirements.txt
```

## Workflow
- LG Gram (weekdays): develop and debug with --fraction 0.02
- MSI RTX 3050 (weekends): full training runs

## Related Work
This project extends the codec-distortion analysis from:
github.com/chinimini532/i2s-vad-research
