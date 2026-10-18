from figure_style import *

hw = load('hardware_ibm_marrakesh.json')
a1 = next(r for r in hw if not r['bleed'] and r['steps'] == 8 and r['fold'] == 1)
a3 = next(r for r in hw if not r['bleed'] and r['steps'] == 8 and r['fold'] == 3)
exact = load('03_noise_results.json')[0]['exact']
zne = 1.5*a1['postselected']-.5*a3['postselected']
zne_se = np.hypot(1.5*a1['post_se'], .5*a3['post_se'])
reduction = 100*(1-abs(a1['postselected']-exact)/abs(a1['raw']-exact))
metrics = {'exact': exact, 'raw': a1['raw'], 'postselected': a1['postselected'],
           'zne': zne, 'zne_se': float(zne_se), 'absolute_error_reduction_percent': reduction,
           'retained_1': round(a1['shots']*a1['kept_fraction']),
           'retained_3': round(a3['shots']*a3['kept_fraction'])}
(OUT/'mitigation_metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
fig, (a, b) = canvas(gap=.48)
values = [a1['raw'], a1['postselected'], zne]
errors = [a1['raw_se'], a1['post_se'], zne_se]
for i, (value, error, color) in enumerate(zip(values, errors, [C['raw'], C['gate'], C['sqa']])):
    a.errorbar(i, value, yerr=error, color=color, marker='o', capsize=4, linestyle='none')
a.axhline(exact, color=C['exact'], linestyle=':', linewidth=1)
a.text(-.18, .117, 'Exact 0.1040', fontsize=8)
a.set(xlim=(-.55, 2.55), ylim=(0, .14), ylabel='Receiver energy fraction',
      xticks=[0, 1, 2], xticklabels=['Raw', 'Post-\nselection', '+ ZNE'],
      yticks=[0, .04, .08, .12])
b.bar([0, 1], [100*a1['kept_fraction'], 100*a3['kept_fraction']],
      color=[C['gate'], C['sqa']], width=.55)
for i, row in enumerate([a1, a3]):
    kept = round(row['shots']*row['kept_fraction'])
    b.text(i, 100*row['kept_fraction']+.7, f'{kept:,} / {row["shots"]:,}',
           ha='center', fontsize=9)
b.set(xticks=[0, 1], xticklabels=['Fold ×1', 'Fold ×3'], ylabel='Shots retained (%)',
      ylim=(0, 22), yticks=[0, 5, 10, 15, 20], xlim=(-.6, 1.6))
save(fig, 'hardware_mitigation')
