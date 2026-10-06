"""Freeze the DEV-selected checkpoint without reading or evaluating FINAL TEST 2."""
import argparse
import copy
import shutil

import yaml

from .experiment_state import experiment_signature, read_json
from .metric_utils import file_digest, write_json
from .phase26_common import dev_only
from .utils import ROOT, load_config, resolve_path


def freeze_selection(config_path=None):
    """Copy verified best weights into a separate immutable selection artifact."""
    path = resolve_path(config_path or 'configs/phase26_selected.yaml')
    with dev_only():
        config = load_config(path)
        state = read_json(config['paths']['training_state_json'])
        choice = copy.deepcopy(config['selection'])
        digest = file_digest(config['paths']['best_checkpoint'])
        if (state['status'] != 'completed' or state['best_checkpoint_sha256'] != digest
                or state['best_epoch'] != choice['best_epoch']
                or abs(state['best_validation_score'] - choice['dev_val_score']) > 1e-12):
            raise ValueError('DEV selection does not match completed training state')
        directory = ROOT / 'outputs/phase26/selected'
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / 'best.pt'
        if target.exists():
            if file_digest(target) != digest:
                raise ValueError('Selection already frozen to another checkpoint; refusing overwrite')
        else:
            shutil.copy2(resolve_path(config['paths']['best_checkpoint']), target)
        if file_digest(target) != digest:
            raise ValueError('Frozen checkpoint copy failed verification')
        write_json(state, directory / 'training_state.json')
        config['paths']['best_checkpoint'] = target.relative_to(ROOT).as_posix()
        config['paths']['training_state_json'] = (directory / 'training_state.json').relative_to(ROOT).as_posix()
        audit = read_json(config['paths']['split_audit_json'])
        choice.update(best_checkpoint_sha256=digest, experiment_signature=experiment_signature(config),
                      dev_partition_sha256={key: audit['partition_sha256'][key] for key in
                          ['metric_train_csv', 'val_candidates_csv', 'val_queries_csv', 'val_ground_truth_csv']},
                      frozen=True)
        choice['final_test2_status'] = 'LOCKED; never used for tuning or evaluated'
        config['selection'] = choice
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
        write_json(choice, ROOT / 'outputs/phase26/selection.json')
        write_json({'config_sha256': file_digest(path), 'best_checkpoint_sha256': digest,
                    'final_test2_evaluations': 0, 'criterion': 'DEV VAL competition score only'},
                   directory / 'selection_frozen.json')
        print(f'Frozen DEV selection: {choice["selected_experiment"]}, best epoch {choice["best_epoch"]}', flush=True)
        return config


def main():
    """Freeze already completed DEV results; this command cannot run final TEST."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/phase26_selected.yaml'))
    freeze_selection(parser.parse_args().config)


if __name__ == '__main__':
    main()
