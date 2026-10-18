from figure_style import *

data = load('audited_simulations.json')
gate = [next(r for r in data['pilot'] if r['method']==name) for name in ['gate_ideal','gate_depolarizing']]
fig, (a, b) = canvas(widths=(.85, 1.15), gap=.5)
a.bar([0, 1], [r['relative_l2_percent'] for r in gate], color=[C['gate'], C['noise']], width=.55)
for i, row in enumerate(gate):
    a.scatter(i+np.array([-.09,0,.09]),row['repeat_error_percent'],s=17,
              color=C['exact'],edgecolor='white',linewidth=.4,zorder=4)
    offset = 1.6 if i == 0 else 1.0
    a.text(i, row['relative_l2_percent']+offset, f'{row["relative_l2_percent"]:.2f}%', ha='center')
a.axhline(1, color=C['exact'], linestyle=':', linewidth=1)
a.text(.45, 1.35, '1% target', fontsize=8)
a.set(xticks=[0, 1], xticklabels=['Ideal', '0.1% gate\nnoise'],
      ylabel='Relative L2\npressure error (%)', ylim=(0, 11), yticks=[0, 5, 10], xlim=(-.6, 1.6))
a.text(.5,1.21,'Bars: mean trace · dots: individual runs',transform=a.transAxes,
       ha='center',fontsize=7.6)
offsets = [-.17, 0, .17]
for key, label, marker, color, offset in zip(['classical','ideal','noisy'],
        ['Classical','Gate, ideal','Gate, noisy'], ['s','o','^'],
        [C['exact'],C['gate'],C['noise']], offsets):
    b.plot([r[key][1] for r in data['inversions']], np.array([1, 0])+offset,
           color=color, marker=marker, linestyle='none', label=label)
b.axvline(1, color=C['raw'], linestyle='--', linewidth=1)
b.axvline(data['inversions'][0]['truth'][1], color=C['exact'], linestyle=':', linewidth=1)
b.text(.999, 1.5, 'Start', ha='center', fontsize=8, color=C['raw'])
b.text(1.1, 1.5, 'Truth', ha='center', fontsize=8)
b.set(xlabel='Second-cell speed ratio', yticks=[0,1],
      yticklabels=['Bipolar pulse','Initial pulse'], ylim=(-.5,1.85),
      xlim=(.95,1.115), xticks=[.96,1,1.05,1.1])
b.legend(loc='upper center', bbox_to_anchor=(.5,-.36), ncol=3, frameon=False,
         fontsize=7.6, handletextpad=.2, columnspacing=.8)
fig.subplots_adjust(bottom=.3)
save(fig, 'pressure_simulations')
