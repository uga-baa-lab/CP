# CHEATCP Kinematic Analysis

## Overview

This project analyzes kinematic data from reaching and interception trials in participants with Cerebral Palsy (CP) and typically developing controls (TDC). The analysis focuses on calculating various kinematic parameters including Initial Movement Direction Error (IDE), Path Length Ratio (PLR), and Path Variability (PVar).

## Key Features

- Processing of raw kinematic data from KINARM trials
- Calculation of key kinematic metrics:
  - Initial Movement Direction Error (IDE)
  - Path Length Ratio (PLR)
  - Path Variability (PVar)
  - End Point Error
  - Reaction Time (RT)
  - Movement Time (MT)
- Regression analysis for subject-level data
- Visualization of hand trajectories and movement paths
- Generation of detailed reports in CSV and Excel formats

## Project Structure

### Main Scripts
- `CHEATCP_Final_With_IE.py`: Core functionality for data processing and analysis
- `InitialEstimateFinalCode.py`: Implementation of initial estimate calculations
- `CP_Main_Final.py`: Main execution script
- `Pvar_final.py`: Path variability calculations

### Support Modules
- `Utils.py`: Utility functions for data processing and analysis
- `Config.py`: Configuration settings and parameters
- `TrialPlots.py`: Visualization functions
- `ReportGenerator.py`: Report generation functionality

## Dependencies

- Python 3.x
- NumPy
- Pandas
- SciPy
- Matplotlib
- Seaborn
- scikit-learn

## Installation

1. Clone the repository
2. Install required packages:

## Usage

1. Configure paths in `Config.py`
2. Run the main analysis:

## Data Requirements

The program expects:
- KINARM trial data in .mat format
- Master data file in Excel format containing subject information

## Output

The analysis generates:
- Processed trial data in CSV format
- Visualization plots of hand trajectories
- Excel reports with kinematic parameters
- Path variability analysis results

## Contributing

Contributions to improve the analysis pipeline are welcome. Please submit pull requests with any enhancements.

## License

This project is licensed under the MIT License.
