"""
Reporting subpackage for CHEAT-CP Kinematic Analysis.

Contains:
  - helpers.py          : Shared utilities (pivot, reindex, save_excel, combine columns)
  - report_generator.py : All 6 report types (means, stds, IDE, PVar)

Usage:
    from cp_analysis.reporting.report_generator import main
    main(data_file_location, report_results_dir, pvar_results_dir, master_df)
"""
