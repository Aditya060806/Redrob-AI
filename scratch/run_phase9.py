import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.phase9_competition_report import generate_competition_report

generate_competition_report('analysis_results')
print("Successfully generated COMPETITION_REPORT.md!")
