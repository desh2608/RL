# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recompute audited pool membership without native workers or model calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    code = Path(__file__).resolve().parent
    if str(code).startswith('/lustre/'):
        parser.error('Copy tools/pivot_pool to node-local disk before importing it.')
    bundle = args.bundle.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='pivot-selection-', dir='/tmp') as temporary:
        reference = Path(temporary) / 'selection_reference'
        shutil.copytree(bundle / 'verifier/selection_reference', reference,
                        ignore=shutil.ignore_patterns('__pycache__'))
        pins = json.loads((reference / 'PINS.json').read_text())
        for name, expected in pins.items():
            actual = hashlib.sha256((reference / name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'Selection reference pin mismatch: {name}')
        os.environ['PIVOT_HOLD_REFERENCE'] = str(reference)
        sys.path.insert(0, str(code))
        from holds70 import decide
        from select_pool import dedupe, limits
        from v3_reference import jlread, jl, read, sha, write
        rows = jlread(bundle / 'MANIFEST.jsonl')
        expected = {r['record_id']: {k: r.get(k) for k in
                    ['duplicate_of', 'in_default_pool', 'not_default_reason']} for r in rows}
        decisions = {r['record_id']: decide(r) for r in rows}
        for row in rows:
            holds = decisions[row['record_id']]
            if not row['audio_only'] and row['passes_eligibility']:
                row.update(duplicate_of=None, in_default_pool=False,
                           not_default_reason='INBOX71:' + ','.join(holds) if holds else None)
        eligible, drops, near = dedupe([r for r in rows if not decisions[r['record_id']]])
        capacities, limit_drops = limits(eligible)
        for row in rows:
            actual = {k: row.get(k) for k in expected[row['record_id']]}
            if actual != expected[row['record_id']]:
                raise ValueError(f'Selection mismatch: {row["record_id"]}')
        generated = {
            'DEDUPE.json': drops, 'NEAR_DUPLICATES.json': near,
            'CAPACITIES.json': capacities, 'LIMIT_DROPS.json': limit_drops,
        }
        for name, value in generated.items():
            if value != read(bundle / name):
                raise ValueError(f'Reproduction mismatch: {name}')
            write(args.out / name, value)
        jl(args.out / 'HOLD_DECISIONS.jsonl',
           [dict(record_id=r['record_id'], holds=decisions[r['record_id']]) for r in rows])
        if sha(args.out / 'HOLD_DECISIONS.jsonl') != sha(bundle / 'HOLD_DECISIONS.jsonl'):
            raise ValueError('Reproduction mismatch: HOLD_DECISIONS.jsonl')
        defaults = [r['record_id'] for r in rows if r['in_default_pool']]
        write(args.out / 'DEFAULT_IDS.json', defaults)
        receipt = dict(status='PASS', records=len(rows), default=len(defaults),
                       all_selection_fields_identical=True,
                       manifest_sha256=sha(bundle / 'MANIFEST.jsonl'),
                       compared_files=list(generated) + ['HOLD_DECISIONS.jsonl'],
                       source_files={p.name: sha(p) for p in sorted(code.glob('*.py'))},
                       native_calls=0, model_calls=0, judge_calls=0, verify_calls=0)
        write(args.out / 'SELECTION_REPRODUCTION.json', receipt)
        print(json.dumps({k: v for k, v in receipt.items() if k != 'source_files'}), flush=True)


if __name__ == '__main__':
    main()
