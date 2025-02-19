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

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Brain-Action-Lab/CHEATCP-Kinematic-Analysis.git
cd CHEATCP-Kinematic-Analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

1. Configure paths in `Config.py`:
```python
BASE_DIR = '/path/to/your/project'
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
```

2. Run the main analysis:
```bash
python CP_Main_Final.py
```

## Data Requirements

The program expects:
- KINARM trial data in .mat format
- Master data file in Excel format containing subject information

Sample data structure is provided in `data/sample/` directory.

## Output

The analysis generates:
- Processed trial data in CSV format
- Visualization plots of hand trajectories
- Excel reports with kinematic parameters
- Path variability analysis results

## Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-measurement`)
3. Commit your changes (`git commit -m 'Add some measurement'`)
4. Push to the branch (`git push origin feature/new-measurement`)
5. Open a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{CHEATCP_Kinematic_Analysis,
  author = {Barany, Deborah and Brain Action Lab},
  title = {CHEATCP Kinematic Analysis},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Brain-Action-Lab/CHEATCP-Kinematic-Analysis}
}
```

## Contact

For questions or feedback, please contact:
- Deborah Barany, Brain and Action Lab, University of Georgia
- dbarany@uga.edu

## Acknowledgments

- Brain and Action Lab members for their contributions
- University of Georgia for supporting this research

