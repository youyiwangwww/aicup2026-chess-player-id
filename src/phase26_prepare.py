"""Archive Round 1 and create a fresh fixed DEV/locked FINAL TEST 2 split."""
import argparse
import hashlib
import shutil

import pandas as pd

from .build_metric_split import THREE_WAY_KEYS, build_three_way_split
from .experiment_state import read_json
from .metric_utils import file_digest, write_json
from .utils import ROOT, load_config, read_csv, resolve_path, write_csv


def preserve_round1(config):
    """Copy Round 1 artifacts once, verifying every original file by SHA-256."""
    archive = resolve_path(config['paths']['round1_archive'])
    receipt = archive / 'archive_manifest.json'
    if receipt.exists():
        manifest = read_json(receipt)
        for path, digest in manifest['files'].items():
            if file_digest(ROOT / path) != digest or file_digest(archive / path) != digest:
                raise ValueError(f'Round 1 preservation mismatch: {path}')
        return
    summary = read_json(config['paths']['round1_summary'])
    if summary['dataset_kind'] != 'real' or summary['final_test_evaluations'] != 1:
        raise ValueError('Round 1 must be a completed real experiment')
    files = []
    for directory in ['outputs/splits', 'outputs/checkpoints', 'outputs/results', 'outputs/features25']:
        files.extend(path for path in (ROOT / directory).rglob('*') if path.is_file())
    files.append(ROOT / 'configs/real_quick.yaml')
    manifest = {}
    for source in files:
        relative = source.relative_to(ROOT)
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[relative.as_posix()] = file_digest(source)
        if file_digest(target) != manifest[relative.as_posix()]:
            raise ValueError('Round 1 archive copy failed')
    write_json({'files': manifest, 'closed_test': True}, receipt)


def prepare(config):
    """Write truth once at split creation; later DEV stages never reopen it."""
    preserve_round1(config)
    path = resolve_path(config['paths']['split_audit_json'])
    if path.exists():
        audit = read_json(path)
        if (audit['source_csv_sha256'] != file_digest(config['paths']['training_csv'])
                or audit['seed'] != config['seed'] or audit['split_settings'] != config['split']):
            raise ValueError('Source changed; do not silently recreate the development split')
        print('Existing fixed Phase 2.6 split retained', flush=True)
        return
    source = read_csv(config['paths']['training_csv'], ['player_id', 'game_id', 'rank', 'color', 'sgf_content'])
    # Closed identities must never become DEV labels, even under a newly sampled split.
    closed = read_csv(config['paths']['round1_test_candidates'], ['player_id', 'game_id'])
    excluded = set(closed.player_id)
    frame = source[~source.player_id.isin(excluded)]
    result = build_three_way_split(frame, **config['split'], seed=config['seed'])
    audit = result[-1]
    audit.update(source_csv_sha256=file_digest(config['paths']['training_csv']),
                 closed_round1_excluded_players=sorted(excluded), final_test2_status='LOCKED',
                 split_settings=config['split'])
    hashes = {}
    for key, partition in zip(THREE_WAY_KEYS, result[:-1]):
        write_csv(partition, config['paths'][key])
        # Derive truth provenance from the generated frame; do not reopen locked truth.
        hashes[key] = hashlib.sha256(partition.to_csv(index=False).encode('utf-8-sig')).hexdigest()
    audit['partition_sha256'] = hashes
    write_json(audit, path)
    print(f'Created fixed DEV split: {audit}', flush=True)


def main():
    """Prepare artifacts without evaluating either TEST."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/phase26.yaml'))
    args = parser.parse_args()
    prepare(load_config(args.config))


if __name__ == '__main__':
    main()
