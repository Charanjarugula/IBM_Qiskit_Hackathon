"""Shared publication plot style. Plotting saved data only; no solver imports."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import svgutils.transform as svg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
C = {'raw': '#7b8790', 'gate': '#0072B2', 'noise': '#D55E00',
     'sqa': '#CC79A7', 'sa': '#009E73', 'exact': '#202833'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.labelsize': 9, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'axes.edgecolor': '#68737c', 'axes.labelcolor': '#202833',
    'text.color': '#202833', 'xtick.color': '#34404a', 'ytick.color': '#34404a',
    'lines.linewidth': 1.3, 'lines.markersize': 4.2, 'pdf.fonttype': 42,
    'svg.fonttype': 'none', 'savefig.dpi': 300, 'figure.facecolor': 'white',
    'axes.facecolor': 'white'})

def load(name):
    return json.loads((ROOT / 'sources' / name).read_text())

def canvas(widths=(1, 1), gap=.42):
    fig, axes = plt.subplots(1, 2, figsize=(183/25.4, 68/25.4),
                             gridspec_kw={'width_ratios': widths})
    fig.subplots_adjust(left=.095, right=.98, top=.79, bottom=.27, wspace=gap)
    for label, ax in zip(('a', 'b'), axes):
        ax.text(-.15, 1.08, label, transform=ax.transAxes, fontsize=11, weight='bold')
        ax.tick_params(length=3, width=.7)
    return fig, axes

def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf')
    fig.savefig(OUT / f'{name}.png', dpi=300)
    fig.savefig(OUT / f'{name}.svg')
    # Preserve inspectable text and an explicit 183 mm physical canvas.
    source = svg.fromfile(str(OUT / f'{name}.svg'))
    composed = svg.SVGFigure('183mm', '68mm')
    composed.root.set('viewBox', '0 0 518.7401574803 192.7559055118')
    composed.append(source.getroot())
    composed.save(str(OUT / f'{name}.svg'))
    plt.close(fig)
    print(name)
