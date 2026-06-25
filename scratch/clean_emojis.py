import os
import re

def clean_file(path):
    print(f"Cleaning: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replacements list
    replacements = [
        # Unicode formatting codes
        ('\uFE0F', ''),
        ('\u200D', ''),
        
        # Arrow and math replacements
        ('—', '-'),
        ('→', '->'),
        ('←', '<-'),
        ('×', 'x'),
        ('±', '+-'),
        ('μ', 'mean'),
        ('σ', 'std'),
        
        # Emojis and checkmarks replacements
        ('🚀', ''),
        ('✅', ''),
        ('❌', ''),
        ('⚠', ''),
        ('⏱️', ''),
        ('↗️', ''),
        ('⭐', ''),
        ('⭐️', ''),
        ('✓', ''),
        ('✗', ''),
        ('🔗', ''),
        ('📂', ''),
        ('🎉', ''),
        ('🏗️', ''),
        ('📦', ''),
        ('⚡', ''),
        ('📊', ''),
        ('📁', ''),
        ('⚡', ''),
        ('├', '|'),
        ('└', '|'),
        ('─', '-'),
        ('│', '|'),
        
        # Remove bracket labels or replace with standard plain text labels
        ('[OK] ', 'OK: '),
        ('[ERROR] ', 'ERROR: '),
        ('[SUCCESS] ', 'SUCCESS: '),
        ('[WARNING] ', 'WARNING: '),
        ('[FAILED] ', 'FAILED: '),
        ('[START] ', 'START: '),
        ('[VALIDATED] ', 'VALIDATED: '),
        ('[SETUP] ', 'SETUP: '),
        
        # Specific comment instructions or placeholders to remove
        ('  # Replace with your actual team name', ''),
        ('     # Replace with your actual email', ''),
        ('                 # Replace with your phone number', ''),
        (' # Replace with your GitHub repo URL', ''),
        (' # Replace with your HF Space link', ''),
        ('# Save this file to your repo root as `submission_metadata.yaml` and fill in any specific contact details.', ''),
    ]
    
    for old, new in replacements:
        content = content.replace(old, new)
        
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    files_to_clean = [
        'README.md',
        'submission_metadata.yaml',
        'test_advanced_model.py',
        'test_pipeline.py',
        'sandbox/app.py',
        'src/shre/stage4_submit.py',
        'src/shre/stage3_ranking.py',
        'src/shre/stage3_ranking_advanced.py',
        'validation/README.md',
        'validation/ablation_study.py',
        'validation/comprehensive_report.py',
        'validation/honeypot_validation.py',
        'validation/learning_curves.py',
        'validation/ranking_metrics.py',
        'validation/validation_runner.py',
        'scratch/test_local.py',
        'scratch/label_tool.py',
        'colab_reproduction.ipynb',
    ]
    
    # Also find files in analysis directory
    for root, dirs, files in os.walk('analysis'):
        for f in files:
            if f.endswith('.py') or f.endswith('.md'):
                files_to_clean.append(os.path.join(root, f))
                
    for path in files_to_clean:
        if os.path.exists(path):
            clean_file(path)
        else:
            print(f"File not found: {path}")

if __name__ == '__main__':
    main()

