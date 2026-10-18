from figure_style import *

hw = [r for r in load('hardware_ibm_marrakesh.json') if not r['bleed'] and r['fold'] == 1]
sim = [r for r in load('03_noise_results.json') if not r['bleed'] and r['fold'] == 1]
assert [r['steps'] for r in hw] == [r['steps'] for r in sim]
steps = [r['steps'] for r in hw]
fig, (a, b) = canvas(widths=(1.2, 1), gap=.43)
a.errorbar(steps, [r['raw'] for r in hw], yerr=[r['raw_se'] for r in hw],
           color=C['raw'], marker='s', capsize=2.5, label='Hardware, raw')
a.errorbar(steps, [r['postselected'] for r in hw], yerr=[r['post_se'] for r in hw],
           color=C['gate'], marker='o', capsize=2.5, label='Hardware, post-selected')
a.errorbar(steps, [r['noisy_post'] for r in sim], yerr=[r['post_se'] for r in sim],
           color=C['noise'], marker='^', linestyle='--', capsize=2.5,
           label='FakeKingston, post-selected')
a.axhline(sim[0]['exact'], color=C['exact'], linestyle=':', linewidth=1)
a.text(2.1, .108, 'Exact reference 0.1040', fontsize=8)
a.set(xlabel='Trotter steps', ylabel='Receiver energy fraction', xticks=steps,
      ylim=(0, .123), yticks=[0, .04, .08, .12], xlim=(1.6, 8.4))
fig.legend(*a.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.53, .98),
           ncol=3, frameon=False, handlelength=1.7, columnspacing=1.2)
b.plot(steps, [100*r['kept_fraction'] for r in hw], color=C['gate'], marker='o')
b.plot(steps, [100*r['kept'] for r in sim], color=C['noise'], marker='^', linestyle='--')
b.set(xlabel='Trotter steps', ylabel='Shots retained (%)', xticks=steps,
      ylim=(0, 80), yticks=[0, 20, 40, 60, 80], xlim=(1.6, 8.4))
b.annotate(f"{100*hw[-1]['kept_fraction']:.1f}%", (8, 100*hw[-1]['kept_fraction']),
           xytext=(-32, -20), textcoords='offset points', color=C['gate'], fontsize=9)
save(fig, 'hardware_sweep')
