from figure_style import *

data = load('audited_simulations.json')
sa = next(r for r in data['pilot'] if r['method']=='sa')
sqa = next(r for r in data['pilot'] if r['method']=='sqa')
rows = [sa, sqa]
fig, (a, b) = canvas(gap=.5)
for ax, key, label, upper in [(a,'relative_l2_percent','Relative L2\npressure error (%)',2.2),
                             (b,'seconds','CPU wall time (s)',2.4)]:
    vals = [r[key] for r in rows]
    ax.bar([0,1], vals, color=[C['sa'],C['sqa']], width=.5)
    for i,value in enumerate(vals):
        repeat_key='repeat_error_percent' if ax is a else 'repeat_seconds'
        ax.scatter(i+np.array([-.09,0,.09]),rows[i][repeat_key],s=17,color=C['exact'],
                   edgecolor='white',linewidth=.4,zorder=4)
        ax.text(i,max(rows[i][repeat_key])+upper*.065,f'{value:.2f}',ha='center',fontsize=10)
    ax.set(xticks=[0,1],xticklabels=['SA','SQA'],
           ylabel=label,ylim=(0,upper),xlim=(-.6,1.6))
a.axhline(1, color=C['exact'], linestyle=':', linewidth=1)
a.text(-.48, 1.05, '1% target', fontsize=8)
fig.text(.5,.94,'Bars: error of mean trace / mean time · dots: three individual runs',
         ha='center',fontsize=8)
save(fig, 'annealing_comparison')
