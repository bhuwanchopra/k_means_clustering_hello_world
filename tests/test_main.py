
import subprocess
import sys

def test_cli_segments_runs():
    result = subprocess.run([sys.executable, 'src/main.py', '--segments', '--num_samples', '100'], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Random people data" in result.stdout
    assert "Columns: Age, Income, Purchase History, Frequency of Purchase" in result.stdout
