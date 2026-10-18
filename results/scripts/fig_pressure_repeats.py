"""Saved-data audit and scientific plot of all three prespecified hardware pairs."""
from figure_style import *
import sys,hashlib
sys.path.insert(0, str(ROOT.parent))
from ibm_hardware.readout import analyze_pilot

folder=ROOT/'sources/pressure_paired_hardware'
load_run=lambda name:json.loads((folder/name).read_text())
assert load_run('completed.json')['status']=='complete'
plan=load_run('pilot_circuits.json');records=plan['records']
assert hashlib.sha256((folder/'pilot_isa.qpy').read_bytes()).hexdigest()==plan['isa_qpy_sha256']
ref=np.array(load_run('reference.json')['ideal_pressure_pa']).ravel()
classical=np.array(load_run('reference.json')['classical_pressure_pa']).ravel()
jobs=[]
for entry in load_run('collected.json'):
    prefix=f'job_{entry["index"]:04d}'
    counts=load_run(prefix+'_counts.json')
    assert len(counts)==12 and all(sum(c.values())==4096 for c in counts)
    recomputed=analyze_pilot(records,counts,4096,(2,2,2))
    saved=load_run(prefix+'_pressure.json')
    assert recomputed==saved and saved['calibration_checks_passed']
    intent=load_run(prefix+'_intent.json')
    assert intent['options']['twirling']['enable_gates']==entry['gate_twirling']
    raw=np.array(saved['raw_pressure_pa']).ravel();corrected=np.array(saved['mitigated']['pressure_pa']).ravel()
    metrics=load_run(prefix+'_metrics.json')
    jobs.append(dict(**entry,raw=raw.tolist(),corrected=corrected.tolist(),
        raw_rmse=float(np.sqrt(np.mean((raw-ref)**2))),
        corrected_rmse=float(np.sqrt(np.mean((corrected-ref)**2))),
        relative_error_vs_classical=float(np.linalg.norm(raw-classical)/np.linalg.norm(classical)),
        charged_seconds=metrics['usage']['qpu_charge_time_seconds'],timestamps=metrics['timestamps']))
arms=[[j for j in jobs if j['gate_twirling']==on] for on in [False,True]]
assert all(len(a)==3 for a in arms)
means=[float(np.mean([j['raw_rmse'] for j in a])) for a in arms]
pairs=[]
for pair in range(3):
    u=next(j for j in jobs if j['pair']==pair and not j['gate_twirling'])
    t=next(j for j in jobs if j['pair']==pair and j['gate_twirling'])
    pairs.append(dict(pair=pair,untwirled=u['raw_rmse'],twirled=t['raw_rmse'],
                      percent_reduction=100*(1-t['raw_rmse']/u['raw_rmse'])))
summary=dict(jobs=jobs,pairs=pairs,mean_raw_rmse=means,
    mean_corrected_rmse=[float(np.mean([j['corrected_rmse'] for j in a])) for a in arms],
    percent_reduction_in_mean_rmse=100*(1-means[1]/means[0]),
    charged_seconds=sum(j['charged_seconds'] for j in jobs),
    audit='All 72 count dictionaries have 4096 shots; derived pressure and calibration match saved reports exactly.',
    scope='Three paired jobs, exploratory pilots excluded; short single-backend campaign, not a speedup claim')
(OUT/'pressure_paired_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,(a,b)=canvas(gap=.48)
for rows,label,color,marker in zip(arms,['No gate twirling','Gate twirling'],[C['raw'],C['gate']],['s','o']):
    values=np.array([j['raw'] for j in rows])
    a.errorbar(ref,values.mean(axis=0),yerr=values.std(axis=0,ddof=1),color=color,
               marker=marker,linestyle='none',capsize=2,label=label)
a.plot([0,1],[0,1],linestyle=':',color=C['exact'],linewidth=1)
a.set(xlabel='Noiseless circuit pressure (Pa)',ylabel='Hardware pressure (Pa)',xlim=(-.05,1.05),ylim=(-.8,1.08))
for pair,offset in zip(pairs,[-.035,0,.035]):
    b.plot(np.array([0,1])+offset,[pair['untwirled'],pair['twirled']],color='#a6b1b8',linewidth=1,zorder=1)
    b.scatter(0+offset,pair['untwirled'],marker='s',color=C['raw'],s=22,zorder=3)
    b.scatter(1+offset,pair['twirled'],marker='o',color=C['gate'],s=22,zorder=3)
for x,value in enumerate(means):
    b.plot([x-.13,x+.13],[value,value],color=C['exact'],linewidth=2,zorder=4)
    b.text(x+.18,value+(.15 if x==0 else .05),f'mean\n{value:.3f}',fontsize=8,va='center')
b.set(xticks=[0,1],xticklabels=['Untwirled','Gate twirling'],ylabel='Pressure RMSE (Pa)',
      xlim=(-.3,1.6),ylim=(0,.95))
b.text(.5,1.18,'Lines: matched pairs · bars: means',ha='center',transform=b.transAxes,fontsize=8)
handles,labels=a.get_legend_handles_labels()
fig.legend(handles,labels,loc='upper left',ncol=2,frameon=False,bbox_to_anchor=(.07,1.01),fontsize=8)
save(fig,'pressure_repeats')
print(json.dumps({k:v for k,v in summary.items() if k not in ['jobs']},indent=2))
