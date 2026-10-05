"""Game-balanced embedding aggregation and candidate/query retrieval."""
import numpy as np
import torch

from .baseline import predict
from .feature_cache import load_store
from .metric_utils import choose_device, file_digest, metric_parser, seed_everything, write_json
from .player_model import PlayerEncoder
from .utils import load_config, resolve_path, write_csv


def embed_groups(model, store, device, batch_size=64):
    """Average positions per game, then games per group; normalize only at the end."""
    if batch_size < 1:
        raise ValueError('inference batch_size must be positive')
    model.eval()
    sums, counts = {}, {}
    with torch.inference_mode():
        for index, record in enumerate(store.records):
            features = store.features(index)
            game_sum = None
            for start in range(0, len(features), batch_size):
                tensor = torch.from_numpy(features[start:start + batch_size].astype(np.float32)).to(device)
                part = model(tensor).cpu().numpy().sum(axis=0, dtype=np.float64)
                game_sum = part if game_sum is None else game_sum + part
            game_mean = game_sum / len(features)
            group = record['group_id']
            sums[group] = sums.get(group, np.zeros_like(game_mean)) + game_mean
            counts[group] = counts.get(group, 0) + 1
    embeddings = {}
    for group in sorted(sums):
        mean = sums[group] / counts[group]
        norm = np.linalg.norm(mean)
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError(f'Nonfinite or zero aggregated embedding: {group}')
        embeddings[group] = mean / norm
    return embeddings


def retrieve(model, candidate_store, query_store, device, batch_size=64, top_k=5):
    """Retrieve Top-k from group embeddings without reading evaluation ground truth."""
    candidates = embed_groups(model, candidate_store, device, batch_size)
    queries = embed_groups(model, query_store, device, batch_size)
    return predict(candidates, queries, top_k), candidates, queries


def load_encoder(checkpoint_path, config, device):
    """Load tensor-only checkpoint data and reject incompatible feature sampling."""
    checkpoint = torch.load(resolve_path(checkpoint_path), map_location='cpu', weights_only=True)
    if (checkpoint['feature_config'] != config['features']
            or checkpoint['feature_seed'] != config['seed']):
        raise ValueError('Checkpoint feature settings/seed mismatch; use its original config')
    if checkpoint['train_source_sha256'] != file_digest(config['paths']['metric_train_csv']):
        raise ValueError('Checkpoint was trained on a different split; retrain or restore the original split')
    model = PlayerEncoder(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def main():
    """Load best checkpoint, retrieve candidate players and save normalized embeddings."""
    parser = metric_parser(__doc__)
    parser.add_argument('--checkpoint', help='Override checkpoint path')
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config['training']
    seed_everything(config['seed'], settings['deterministic'], settings['num_threads'])
    device = choose_device(settings['device'])
    model, checkpoint = load_encoder(args.checkpoint or config['paths']['best_checkpoint'], config, device)
    candidates = load_store(config, 'candidate')
    queries = load_store(config, 'query')
    predictions, candidate_embeddings, query_embeddings = retrieve(
        model, candidates, queries, device, config['inference']['batch_size'], config['inference']['top_k'])
    write_csv(predictions, config['paths']['predictions_csv'])
    prediction_path = resolve_path(config['paths']['predictions_csv'])
    write_json({'prediction_sha256': file_digest(prediction_path),
                'candidate_source_sha256': candidates.manifest['source_sha256'],
                'query_source_sha256': queries.manifest['source_sha256'],
                'checkpoint_sha256': file_digest(args.checkpoint or config['paths']['best_checkpoint']),
                'checkpoint_epoch': checkpoint['epoch']}, prediction_path.with_suffix('.metadata.json'))
    output = resolve_path(config['paths']['embeddings_npz'])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, candidate_ids=np.array(list(candidate_embeddings)),
                        candidate_embeddings=np.stack(list(candidate_embeddings.values())),
                        question_ids=np.array(list(query_embeddings)),
                        query_embeddings=np.stack(list(query_embeddings.values())))
    print(f'checkpoint epoch={checkpoint["epoch"]}, device={device}, '
          f'candidates={len(candidate_embeddings)}, questions={len(query_embeddings)}, '
          f'prediction rows={len(predictions)}', flush=True)


if __name__ == '__main__':
    main()
