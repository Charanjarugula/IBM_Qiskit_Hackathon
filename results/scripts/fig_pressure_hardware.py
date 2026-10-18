"""Plot the two exploratory, fixed-circuit pressure hardware pilots."""
from figure_style import *

names=['pressure_pilot_hardware','pressure_pilot_twirled']
rows=[]
for name in names:
    folder=ROOT/'sources'/name
    report=json.loads((folder/'pilot_pressure.json').read_text())
    ref=json.loads((folder/'reference.json').read_text())
    target=np.array(ref['ideal_pressure_pa']).ravel()
    raw=np.array(report['raw_pressure_pa']).ravel()
    corrected=np.array(report['mitigated']['pressure_pa']).ravel() if report['mitigated'] else None
    se=np.array(report['raw_standard_error_pa']).ravel()
    rmse=lambda p:float(np.sqrt(np.mean((p-target)**2)))
    rows.append(dict(name=name,reference=target.tolist(),raw=raw.tolist(),se=se.tolist(),
        corrected=None if corrected is None else corrected.tolist(),raw_rmse=rmse(raw),
        corrected_rmse=None if corrected is None else rmse(corrected),
        job_id=json.loads((folder/'job_0000_submitted.json').read_text())['job_id'],
        charged_seconds=json.loads((folder/'job_0000_metrics.json').read_text())['usage']['qpu_charge_time_seconds'],
        calibration_passed=report['calibration_checks_passed']))
assert rows[0]['reference']==rows[1]['reference']
fig,(a,b)=canvas(gap=.48)
for row,label,color,marker in zip(rows,['No gate twirling','Gate twirling'],[C['raw'],C['gate']],['s','o']):
    a.errorbar(row['reference'],row['raw'],yerr=row['se'],linestyle='none',
               marker=marker,color=color,capsize=2,label=label)
    b.plot([0,1],[row['raw_rmse'],row['corrected_rmse']],color=color,marker=marker,label=label)
    for i,v in enumerate([row['raw_rmse'],row['corrected_rmse']]):
        if v is not None:b.annotate(f'{v:.3f}',(i,v),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
a.plot([0,1],[0,1],color=C['exact'],linestyle=':',linewidth=1)
a.set(xlabel='Noiseless circuit pressure (Pa)',ylabel='Hardware pressure (Pa)',
      xlim=(-.05,1.05),ylim=(-.65,1.08))
b.set(xticks=[0,1],xticklabels=['Raw','Readout\ncorrected'],ylabel='Pressure RMSE (Pa)',
      xlim=(-.38,1.38),ylim=(0,max(r['corrected_rmse'] or r['raw_rmse'] for r in rows)*1.25))
handles,labels=a.get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.55,1.01))
save(fig,'pressure_hardware')
(OUT/'pressure_hardware_metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
