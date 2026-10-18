"""Prepare a synthetic 2D/3D layered-head input, not an anatomical MIDA reproduction.

No solver is run by this command. Still requires explicit execution authorization.
Uses Anik's layered phantom; no external dataset is silently downloaded.
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimension', type=int, choices=[2, 3], default=3)
    parser.add_argument('--head-cells', type=int, default=64)
    parser.add_argument('--water-cells', type=int, default=16)
    parser.add_argument('--transducers', type=int, default=32)
    parser.add_argument('--spacing-m', type=float, default=0.003)
    parser.add_argument('--frequency-hz', type=float, default=60000.)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('No input generated. --execute required when authorized.')
    import numpy as np
    from pathlib import Path
    from shared.phantom import head_phantom, LABELS, SPEEDS
    if args.head_cells < 16 or args.water_cells < 4 or args.transducers < 4 or args.spacing_m <= 0 or args.frequency_hz <= 0:
        raise ValueError('Invalid geometry or pulse parameters')
    if Path(args.output).exists():
        raise FileExistsError('Choose a new output path')
    if Path(args.output).suffix != '.npz':
        raise ValueError('Output filename must end in .npz')
    shape = (args.head_cells,)*args.dimension
    centre = np.array(shape)/2
    bleed = tuple((centre + np.array([0]*(args.dimension-1)+[args.head_cells//6])).astype(int))
    c, labels = head_phantom(shape, bleed=bleed, bleed_radius=max(2, args.head_cells//16), normalise=False)
    c = np.pad(c, args.water_cells, constant_values=SPEEDS['water'])
    labels = np.pad(labels, args.water_cells, constant_values=LABELS['water'])
    initial = np.where(labels == LABELS['blood'], SPEEDS['brain'], c)
    n = args.transducers
    if args.dimension == 2:
        angles = np.arange(n)*2*np.pi/n
        directions = np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        z = 1-2*(np.arange(n)+0.5)/n
        angles = np.arange(n)*np.pi*(3-np.sqrt(5))
        radius = np.sqrt(1-z*z)
        directions = np.column_stack([radius*np.cos(angles), radius*np.sin(angles), z])
    centre = (np.array(c.shape)-1)/2
    radius = args.head_cells/2 + args.water_cells/2
    locations = (centre + radius*directions)*args.spacing_m
    dt = 0.2*args.spacing_m/(float(c.max())*np.sqrt(args.dimension))
    # Low-order inputs need adequate points per wavelength even with higher-order Stride.
    if float(c.min())/(args.frequency_hz*args.spacing_m) < 6:
        raise ValueError('Use finer spacing or a lower pulse frequency (at least six cells/wavelength)')
    recording = 2.5*max(c.shape)*args.spacing_m/float(c.min())
    nt = int(np.ceil(recording/dt))
    t = np.arange(nt)*dt
    cycles = 3
    duration = cycles/args.frequency_hz
    wavelet = np.where(t <= duration, np.sin(2*np.pi*args.frequency_hz*t)*np.sin(np.pi*t/duration)**2, 0)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, vp_m_s=c.astype(np.float32), initial_vp_m_s=initial.astype(np.float32),
             spacing_m=np.full(args.dimension, args.spacing_m), dt_s=dt,
             locations_m=locations, source_ids=np.arange(n),
             wavelets=np.tile(wavelet.astype(np.float32), (n, 1)),
             provenance='Synthetic layered phantom, known skull initialization; not MIDA or clinical data')


if __name__ == '__main__':
    main()
