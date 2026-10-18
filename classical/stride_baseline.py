"""Adapted from Stride's breast2D/breast3D examples at commit 8733ecb9 (AGPL-3.0).

Uses upstream forward(), adjoint(), L2DistanceLoss and GradientDescent rather than
reimplementing the acoustic PDE or adjoint. Dataset convention is documented in README.md.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Classical Stride 2D/3D acoustic FWI')
    parser.add_argument('mode', choices=['forward', 'invert'])
    parser.add_argument('--input', required=True, help='Input NPZ following the documented schema')
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('No execution performed. --execute required when authorized.')
    # Keep this local even on a machine with unrelated cloud artifact configuration.
    if any(k.startswith('MOSAIC_ARTIFACT_') for k in os.environ):
        parser.error('Unset MOSAIC_ARTIFACT_* configuration for this local-only baseline')
    import numpy as np
    import mosaic
    from stride import (Space, Time, Problem, ScalarField, Shot, IsoAcousticDevito,
                        L2DistanceLoss, GradientDescent, ProcessGlobalGradient,
                        ProcessModelIteration, OptimisationLoop, forward, adjoint)
    from shared.evidence import Evidence

    cfg = json.loads(Path(args.config).read_text())
    for key in ('padding_cells', 'absorbing_cells', 'iterations_per_frequency'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('step_size_m_s', 'speed_min_m_s', 'speed_max_m_s'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if cfg['speed_min_m_s'] >= cfg['speed_max_m_s']:
        raise ValueError('Invalid speed bounds')
    frequencies = np.asarray(cfg['max_frequency_hz'], dtype=float)
    if frequencies.ndim != 1 or not frequencies.size or not np.isfinite(frequencies).all() or np.any(frequencies <= 0) or np.any(np.diff(frequencies) <= 0):
        raise ValueError('Frequency stages must be positive, finite and increasing')
    with np.load(args.input, allow_pickle=False) as source:
        data = {k: source[k].copy() for k in source.files}
    vp_input = np.asarray(data['vp_m_s'], dtype=np.float32)
    if vp_input.ndim not in (2, 3) or not np.isfinite(vp_input).all() or np.any(vp_input <= 0):
        raise ValueError('vp_m_s must be a positive finite 2D/3D array')
    nd = vp_input.ndim
    spacing = tuple(float(v) for v in data['spacing_m'])
    dt = float(data['dt_s'])
    locations = np.asarray(data['locations_m'], dtype=float)
    if data['source_ids'].dtype.kind not in 'iu':
        raise ValueError('source_ids must contain integers')
    source_ids = np.asarray(data['source_ids'], dtype=int)
    wavelets = np.asarray(data['wavelets'], dtype=np.float32)
    if len(spacing) != nd or not np.isfinite(spacing).all() or min(spacing) <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Invalid space/time sampling')
    if locations.ndim != 2 or locations.shape[1] != nd or not np.isfinite(locations).all():
        raise ValueError('locations_m must contain finite coordinates for each transducer')
    limit = (np.array(vp_input.shape)-1)*spacing
    if np.any(locations < 0) or np.any(locations > limit):
        raise ValueError('Transducer outside physical domain')
    if source_ids.ndim != 1 or not len(source_ids) or len(set(source_ids)) != len(source_ids) or np.any(source_ids < 0) or np.any(source_ids >= len(locations)):
        raise ValueError('Invalid source IDs')
    if wavelets.ndim != 2 or wavelets.shape[0] != len(source_ids) or not np.isfinite(wavelets).all():
        raise ValueError('wavelets shape must be (sources, time)')
    nt = wavelets.shape[1]
    if nt < 4 or max(cfg['max_frequency_hz']) >= 0.5/dt:
        raise ValueError('Insufficient time samples or frequency above Nyquist')
    if int(cfg['absorbing_cells']) >= int(cfg['padding_cells']):
        raise ValueError('Absorbing layer must fit inside extra padding')
    # This is a conservative input check, not a substitute for Devito stability validation.
    maximum_speed = max(float(vp_input.max()), cfg['speed_max_m_s']) if args.mode == 'invert' else float(vp_input.max())
    if dt*maximum_speed*np.sqrt(sum(1/h**2 for h in spacing)) >= 0.5:
        raise ValueError('Reduce dt: conservative acoustic Courant check failed')
    if args.mode == 'invert' and (np.any(vp_input < cfg['speed_min_m_s']) or np.any(vp_input > cfg['speed_max_m_s'])):
        raise ValueError('Initial speed model lies outside inversion bounds')
    if 'initial_vp_m_s' in data:
        initial = data['initial_vp_m_s']
        if initial.shape != vp_input.shape or not np.isfinite(initial).all() or np.any(initial <= 0):
            raise ValueError('initial_vp_m_s must be positive, finite and match vp shape')
    expected = (len(source_ids), len(locations), nt)
    if args.mode == 'invert' and ('observed_pressure' not in data or data['observed_pressure'].shape != expected or not np.isfinite(data['observed_pressure']).all()):
        raise ValueError(f'Inversion needs finite observed_pressure with shape {expected}')
    if 'rho_kg_m3' in data and (data['rho_kg_m3'].shape != vp_input.shape or np.any(data['rho_kg_m3'] <= 0) or not np.isfinite(data['rho_kg_m3']).all()):
        raise ValueError('Density must be positive, finite and match vp shape')
    evidence = Evidence(args.output, args.config, 'stride_'+args.mode)
    import hashlib
    evidence.write('input.json', {'sha256': hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                                 'shape': list(vp_input.shape), 'sources': len(source_ids),
                                 'receivers': len(locations), 'samples': nt,
                                 'density': 'supplied_fixed' if 'rho_kg_m3' in data else 'constant',
                                 'attenuation': 'none', 'boundary': 'Stride absorbing layer'})

    async def run(runtime):
        space = Space(shape=vp_input.shape, spacing=spacing,
                      extra=(int(cfg['padding_cells']),)*nd,
                      absorbing=(int(cfg['absorbing_cells']),)*nd)
        time = Time(start=0., step=dt, num=nt)
        problem = Problem(name='acoustic_baseline', space=space, time=time,
                          output_folder=str(evidence.path))
        if args.mode == 'invert':
            vp = ScalarField.parameter(name='vp', grid=problem.grid, needs_grad=True)
        else:
            vp = ScalarField(name='vp', grid=problem.grid)
        vp.data[:] = vp_input
        vp.pad()
        problem.medium.add(vp)
        physical_kwargs = {'platform': 'cpu', 'kernel': 'OT2'}
        if 'rho_kg_m3' in data:
            rho = ScalarField(name='rho', grid=problem.grid)
            rho.data[:] = data['rho_kg_m3']
            rho.pad()
            problem.medium.add(rho)
            physical_kwargs['rho'] = rho
        problem.transducers.default()
        for i, position in enumerate(locations):
            problem.geometry.add(i, problem.transducers.get(0), position)
        loc = problem.geometry.locations
        for si, source_id in enumerate(source_ids):
            shot = Shot(si, sources=[loc[int(source_id)]], receivers=loc,
                        geometry=problem.geometry, problem=problem)
            shot.wavelets.data[0, :] = wavelets[si]
            if args.mode == 'invert':
                shot.observed.data[:] = data['observed_pressure'][si]
            problem.acquisitions.add(shot)
        pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)
        if args.mode == 'forward':
            await forward(problem, pde, vp, dump=False, safe=False,
                          upload_acquisitions=False, **physical_kwargs)
            output = dict(data)
            output['observed_pressure'] = np.stack([s.observed.data for s in problem.acquisitions.shots])
            # Forward input may include an initial model to make a ready inversion dataset.
            if 'initial_vp_m_s' in output:
                output['vp_m_s'] = output.pop('initial_vp_m_s')
            output['provenance'] = np.array('Stride synthetic observations; not clinical data')
            np.savez(evidence.path/'inversion_input.npz', **output)
        else:
            loss = L2DistanceLoss.remote(len=runtime.num_workers)
            optimiser = GradientDescent(vp, step_size=float(cfg['step_size_m_s']),
                process_grad=ProcessGlobalGradient(),
                process_model=ProcessModelIteration(min=float(cfg['speed_min_m_s']),
                                                    max=float(cfg['speed_max_m_s'])))
            loop = OptimisationLoop()
            frequencies = cfg['max_frequency_hz']
            for block, freq in loop.blocks(len(frequencies), frequencies):
                await adjoint(problem, pde, loss, loop, optimiser, vp,
                    num_iters=int(cfg['iterations_per_frequency']),
                    select_shots=dict(num=len(source_ids), randomly=False),
                    f_max=float(freq), max_freqs=frequencies, dump=True, safe=False,
                    # Preserve pressure amplitudes; disable optional upstream transforms.
                    differentiate_traces=False, norm_per_shot=False, norm_per_trace=False,
                    mute_first_arrival=False, mute_traces=False,
                    time_tweaking=False, time_weighting=False, **physical_kwargs)
            np.save(evidence.path/'reconstructed_speed_m_s.npy', vp.data)
            from shared.interpretation import interpret
            evidence.write('illustrative_tissue_labels.json', interpret(vp.data))
        evidence.finish({'mode': args.mode, 'backend': 'Stride/Devito CPU',
                         'method': 'upstream forward and discrete-adjoint FWI'})
    try:
        mosaic.run(run)
    except BaseException as error:
        evidence.write('failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise


if __name__ == '__main__':
    main()
