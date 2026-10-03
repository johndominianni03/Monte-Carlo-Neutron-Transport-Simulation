"""Phase 4: benchmark of mcslab against OpenMC 0.16.0 (docs/phase4_plan.md).

    run_mcslab.py   mcslab side (mcslab's venv); writes problems.json and
                    results/mcslab_*.json
    run_openmc.py   OpenMC side (OpenMC's own environment only); reads
                    problems.json, writes results/openmc_*.json
    compare.py      applies the pre-declared checks to a committed pair of
                    results (numpy only)
    jsonio.py       the JSON writer both sides share (standard library only)

The two sides exchange data through these files only: mcslab never imports
openmc, and the OpenMC side never imports mcslab.
"""
