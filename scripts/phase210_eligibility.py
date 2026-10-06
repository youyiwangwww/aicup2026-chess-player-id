"""Count genuinely unseen identities without opening any historical TEST input."""
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metric_utils import file_digest, write_json


def historical_groups():
    """Recover closed cohorts from copied metadata and TRAIN IDs from ID-only reads."""
    stability = json.loads((ROOT/'outputs/phase29/splits/audit.json').read_text(encoding='utf-8'))
    groups = {k: set(v) for k, v in stability['exclusions'].items()}
    groups['stability'] = set(stability['player_ids'])
    provenance = {}
    paths = ['outputs/splits/train.csv', 'outputs/splits/val_candidates.csv',
             'outputs/phase26/splits/dev_train.csv']
    paths += [str(p.relative_to(ROOT)).replace('\\', '/') for p in
              sorted((ROOT/'outputs/phase26/experiments').glob('*/train.csv'))]
    for path in paths:
        groups[path] = set(pd.read_csv(ROOT/path, dtype=str, usecols=['player_id']).player_id)
        provenance[path] = file_digest(path)
    return groups, provenance


def main():
    """Stop before creating a TEST if fewer than 100 unseen players have 40 games."""
    groups, provenance = historical_groups()
    excluded = set().union(*groups.values())
    source = pd.read_csv(ROOT/'data/training/train_A.csv', dtype=str, usecols=['player_id'])
    counts = source.loc[~source.player_id.isin(excluded)].groupby('player_id').size()
    result = {'unseen_players': len(counts), 'maximum_eligible_unseen_players': int((counts >= 40).sum()),
              'historical_unique_players': len(excluded), 'groups': {k: sorted(v) for k, v in groups.items()},
              'identity_source_sha256': provenance,
              'dataset_sha256': file_digest('data/training/train_A.csv')}
    write_json(result, ROOT/'outputs/phase210/eligibility.json')
    print({k: v for k, v in result.items() if k not in ['groups', 'identity_source_sha256']})
    if result['maximum_eligible_unseen_players'] < 100:
        print(f"maximum eligible unseen players = {result['maximum_eligible_unseen_players']}; TEST not created")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
