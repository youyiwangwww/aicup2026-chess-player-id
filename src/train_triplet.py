"""Train only on metric-training players; select best by held-out retrieval score."""
from pathlib import Path
import tempfile
import time
import uuid

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .build_metric_split import is_three_way, read_training_validation, verify_partition_hashes
from .embed_players import retrieve
from .evaluate import evaluate
from .feature_cache import load_store
from .metric_utils import choose_device, file_digest, metric_parser, seed_everything, write_json
from .player_model import PlayerEncoder
from .triplet_dataset import TripletDataset
from .utils import load_config, resolve_path, write_csv


def save_checkpoint(checkpoint, path):
    """Atomically save a complete checkpoint after each epoch."""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.pt', delete=False) as stream:
        temporary = Path(stream.name)
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train(config):
    """Run gradient updates on train data, then read truth only for validation metrics."""
    settings = config['training']
    if min(settings['epochs'], settings['batch_size'], settings['learning_rate'], settings['margin']) <= 0:
        raise ValueError('epochs, batch_size, learning_rate and margin must be positive')
    expected_channels = 2 * config['features']['history_length'] + 1
    if config['model']['in_channels'] != expected_channels:
        raise ValueError('model.in_channels must equal 2*features.history_length+1')
    seed_everything(config['seed'], settings['deterministic'], settings['num_threads'])
    device = choose_device(settings['device'])
    three_way = is_three_way(config)
    state = None
    if three_way:
        from .experiment_state import experiment_signature

        # Mark running before reading data; an interrupted retrain cannot expose old test results.
        state = {'run_id': uuid.uuid4().hex, 'status': 'running', 'seed': config['seed'],
                 'experiment_signature': experiment_signature(config)}
        write_json(state, config['paths']['training_state_json'])
    # This reader never opens TEST data; all validation inputs are independently audited.
    _, _, _, truth = read_training_validation(config)
    train_store = load_store(config, 'train')
    candidate_store = load_store(config, 'val_candidate' if three_way else 'candidate')
    query_store = load_store(config, 'val_query' if three_way else 'query', allow_empty=True)
    dataset = TripletDataset(train_store, settings['samples_per_epoch'], config['seed'])
    if dataset.excluded_players:
        print(f'Excluded training players with <2 valid games: {dataset.excluded_players}', flush=True)
    model = PlayerEncoder(**config['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                                  weight_decay=settings['weight_decay'])
    criterion = nn.TripletMarginLoss(margin=settings['margin'], p=2)
    best_score = float('-inf')
    best_epoch = None
    history = []
    print(f'device={device}; train players={len(dataset.players)}; '
          f'train games={len(train_store)}; triplets/epoch={len(dataset)}; '
          f'parameters={sum(p.numel() for p in model.parameters())}', flush=True)
    for epoch in range(1, settings['epochs'] + 1):
        started = time.perf_counter()
        dataset.set_epoch(epoch - 1)
        generator = torch.Generator().manual_seed(config['seed'] + epoch)
        loader = DataLoader(dataset, batch_size=settings['batch_size'], shuffle=False,
                            num_workers=settings['num_workers'], generator=generator,
                            pin_memory=device.type == 'cuda')
        model.train()
        total_loss, samples = 0.0, 0
        for anchor, positive, negative in loader:
            size = len(anchor)
            tensor = torch.cat([anchor, positive, negative]).to(device)
            optimizer.zero_grad(set_to_none=True)
            a, p, n = model(tensor).split(size)
            loss = criterion(a, p, n)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite triplet loss')
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * size
            samples += size
        # Validation runs in eval/inference mode; no evaluation gradients or labels in loss.
        predictions, _, _ = retrieve(model, candidate_store, query_store, device,
                                     config['inference']['batch_size'], config['inference']['top_k'])
        metrics, _ = evaluate(predictions, truth)
        elapsed = time.perf_counter() - started
        score = metrics['competition_score']
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
        row = {'epoch': epoch, 'train_loss': total_loss / samples,
               'learning_rate': optimizer.param_groups[0]['lr'], 'elapsed_time': elapsed,
               **{f'val_{key}': value for key, value in metrics.items()}, 'is_best': improved}
        row.update(val_top1=metrics['top_1_accuracy'], val_top3=metrics['top_3_accuracy'],
                   val_top5=metrics['top_5_accuracy'])
        history.append(row)
        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'model_config': config['model'], 'feature_config': config['features'],
                      'feature_seed': config['seed'], 'validation_metrics': metrics,
                      'best_score': best_score, 'training_config': settings,
                      'train_source_sha256': train_store.manifest['source_sha256'],
                      'candidate_source_sha256': candidate_store.manifest['source_sha256'],
                      'query_source_sha256': query_store.manifest['source_sha256']}
        if three_way:
            checkpoint['experiment_run_id'] = state['run_id']
            checkpoint['validation_ground_truth_sha256'] = file_digest(config['paths']['val_ground_truth_csv'])
        save_checkpoint(checkpoint, config['paths']['last_checkpoint'])
        if improved:
            save_checkpoint(checkpoint, config['paths']['best_checkpoint'])
        write_csv(pd.DataFrame(history), config['paths']['training_log_csv'])
        print(f'epoch={epoch} train_loss={row["train_loss"]:.6f} '
              f'learning_rate={row["learning_rate"]:.6g} elapsed_time={elapsed:.2f}s '
              f'val_competition_score={score:.6f} best={improved}', flush=True)
    if three_way:
        keys = ['metric_train_csv', 'val_candidates_csv', 'val_queries_csv', 'val_ground_truth_csv']
        audit = verify_partition_hashes(config, keys)
        state.update(status='completed', best_epoch=best_epoch, best_validation_score=best_score,
                     best_checkpoint_sha256=file_digest(config['paths']['best_checkpoint']),
                     partition_sha256={key: audit['partition_sha256'][key] for key in keys},
                     training_elapsed_seconds=sum(row['elapsed_time'] for row in history))
        write_json(state, config['paths']['training_state_json'])
    return history


def main():
    """Train using YAML settings, automatically selecting CUDA or CPU."""
    args = metric_parser(__doc__).parse_args()
    train(load_config(args.config))


if __name__ == '__main__':
    main()
