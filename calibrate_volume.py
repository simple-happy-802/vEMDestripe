#!/usr/bin/env python3
"""Save vEMDestripe's whole-volume reference-calibrated final output.

Example: python calibrate_volume.py --input-dir outputs/network_raw \
    --reference-dir screened_original_references --output-dir outputs/final

Use every saved output slice, not a scoring crop. Real-data targets are
original reference snapshots; a paired clean TIFF is accepted only through
the explicit synthetic-analysis option --target-clean-volume.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import tifffile

from vemdestripe.calibration import calibrate_directory, reference_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--input-pattern', default='*_deflickered.png')
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--reference-dir', type=Path,
                        help='Original, unprocessed reference snapshots, not reference.png')
    target.add_argument('--target-clean-volume', type=Path,
                        help='Explicit synthetic-only full clean TIFF target; privileged calibration, not deployment')
    parser.add_argument('--reference-metadata', type=Path,
                        help='processing_info.json: use exactly its reference_indices filenames')
    parser.add_argument('--output-stack', type=Path,
                        help='Additionally export the saved calibrated images as one TIFF stack')
    args = parser.parse_args()
    if args.reference_metadata and not args.reference_dir:
        parser.error('--reference-metadata requires --reference-dir')
    if args.target_clean_volume:
        with tifffile.TiffFile(args.target_clean_volume) as clean:
            def pages():
                for page in clean.pages:
                    value = page.asarray()
                    if value.ndim == 2:
                        yield value
                    elif value.ndim == 3:
                        yield from value
                    else:
                        raise ValueError('Clean target must be a grayscale ZYX TIFF')
            stats = reference_statistics(pages())
        stats['target_available_at_deployment'] = False
        record = calibrate_directory(args.input_dir, args.output_dir,
            input_pattern=args.input_pattern, target_statistics=stats,
            target_source='paired_clean_full_volume_explicit_synthetic_analysis')
    else:
        if args.reference_metadata:
            metadata = json.loads(args.reference_metadata.read_text(encoding='utf-8'))
            names = metadata['reference_indices']
            if not names or any(not isinstance(name, str) or Path(name).name != name for name in names):
                raise ValueError('reference_indices must list the selected original snapshot basenames')
            paths = [args.reference_dir / name for name in names]
        else:
            extensions = {'.png', '.tif', '.tiff', '.jpg', '.jpeg'}
            paths = sorted(p for p in args.reference_dir.iterdir() if p.suffix.lower() in extensions)
        record = calibrate_directory(args.input_dir, args.output_dir, paths,
                                     input_pattern=args.input_pattern)
    if args.output_stack:
        destination = args.output_stack.resolve()
        if destination.exists():
            raise FileExistsError('Choose a new output stack filename')
        from vemdestripe.calibration import read_uint8_image
        paths = sorted(args.output_dir.glob(args.input_pattern))
        destination.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(destination, data=(read_uint8_image(p) for p in paths),
            shape=(len(paths), *record['source_statistics']['shape_yx']), dtype='uint8',
            bigtiff=True, photometric='minisblack', metadata={'axes': 'ZYX'}, contiguous=True)
        destination.with_suffix('.calibration.json').write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({key: record[key] for key in ('scale', 'bias', 'source_statistics', 'target_source', 'complete')}, indent=2))


if __name__ == '__main__':
    main()
