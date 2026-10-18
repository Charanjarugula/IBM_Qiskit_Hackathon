"""Verify archived results and redraw figures. Never connects to IBM or runs solvers."""
from pathlib import Path
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Only NumPy is needed; do not redraw figures.')
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent/'scripts'
    subprocess.run([sys.executable, str(scripts/'verify.py')], check=True)
    if not args.verify_only:
        for name in ['fig_pressure.py', 'fig_hardware.py', 'fig_pressure_repeats.py',
                     'fig_annealing.py', 'fig_mitigation.py', 'fig_speed.py', 'fig_pressure_hardware.py']:
            subprocess.run([sys.executable, str(scripts/name)], check=True)
    print('Saved evidence verified. No simulations, network calls, or hardware submissions performed.')


if __name__ == '__main__':
    main()
