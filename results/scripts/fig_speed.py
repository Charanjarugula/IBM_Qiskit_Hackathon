"""Recorded CPU runtime at the shared pilot configuration, not a speedup projection."""
from figure_style import *

audit = json.loads((ROOT/'verification.json').read_text())
methods = ['classical_exact', 'gate_ideal', 'gate_depolarizing', 'sa', 'sqa']
labels = ['Classical\nexact', 'Gate\nideal Aer', 'Gate\nnoisy Aer', 'SA', 'SQA']
colors = [C['exact'], C['gate'], C['noise'], C['sa'], C['sqa']]
fig, ax = plt.subplots(figsize=(183/25.4, 68/25.4))
fig.subplots_adjust(left=.13, right=.98, bottom=.28, top=.81)
summary = []
for x, (method, color) in enumerate(zip(methods, colors)):
    times = [t for row in audit['timings'] if row['method']==method for t in row['repeat_seconds']]
    mean = float(np.mean(times))
    ax.scatter(x+np.linspace(-.08,.08,len(times)), times, color=color, s=20, zorder=3)
    ax.plot([x-.17,x+.17], [mean,mean], color=color, linewidth=1.7)
    label = f'{mean*1000:.2f} ms' if mean < .01 else f'{mean:.3f} s'
    ax.text(x, max(times)*1.55, label, ha='center', fontsize=9)
    summary.append(dict(method=method, repeats=len(times), seconds=times, mean_seconds=mean))
ax.set(yscale='log', ylabel='CPU wall time per forward run (s)',
       xticks=range(5), xticklabels=labels, xlim=(-.5,4.5), ylim=(.001,5))
ax.set_yticks([.001,.01,.1,1], labels=['0.001','0.01','0.1','1'])
ax.text(.5,1.15,'Recorded pilot timings · dots: repeats · horizontal marks: means',
        transform=ax.transAxes, ha='center', fontsize=8)
ax.text(.5,-.47,'Same reduced task; configured budgets, not matched achieved accuracy',
        transform=ax.transAxes, ha='center', fontsize=8)
save(fig,'runtime_comparison')
(OUT/'runtime_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
