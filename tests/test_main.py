
import subprocess
import sys



def test_cli_runs():
    result = subprocess.run([
        sys.executable, 'src/main.py', '--num_samples', '100'
    ], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Random people data" in result.stdout
    assert "Columns: Age, Income, Purchase History, Frequency of Purchase" in result.stdout

def test_cli_custom_segments():
    result = subprocess.run([
        sys.executable, 'src/main.py',
        '--num_samples', '100',
        '--young_age_min', '20', '--young_age_max', '30', '--young_income_min', '50000', '--young_freq_min', '10',
        '--budget_income_max', '30000', '--budget_freq_min', '5',
        '--loyal_freq_min', '60', '--loyal_purchase_min', '10000'
    ], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Random people data" in result.stdout
    assert "Columns: Age, Income, Purchase History, Frequency of Purchase" in result.stdout
