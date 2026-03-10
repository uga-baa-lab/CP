"""
CHEAT-CP Kinematic Analysis Package
====================================

This package provides a modular pipeline for analyzing upper-limb kinematic data
from KINARM robot experiments, specifically for comparing motor control in
children with Cerebral Palsy (CP) vs Typically Developing Children (TDC).

The analysis covers two task types:
  - **Reaching**: Moving the hand to a stationary target
  - **Interception**: Moving the hand to intercept a moving target

Package Structure:
  - config.py             : Configuration (paths, defaults) — no side effects
  - data_loader.py        : Loading .mat files and Master Excel
  - signal_processing.py  : Low-pass filtering, hand speed, cursor positions
  - kinematics.py         : All kinematic measures (RT, CT, IDE, PLR, etc.)
  - regression.py         : RANSAC regression & Initial Estimate (IE)
  - pvar.py               : Path Variability (PVar) computation
  - pipeline.py           : Orchestrates the full analysis flow
  - plotting/             : All visualization functions
  - reporting/            : Excel report generation

Author: Brain and Action Lab, University of Georgia
"""

__version__ = "2.0.0"
