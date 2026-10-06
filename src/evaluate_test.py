"""Run one final held-out retrieval evaluation after training has completed."""
from .embed_players import load_encoder, retrieve
from .evaluate import evaluate
from .experiment_state import (experiment_parser, read_json, require_training_complete,
                               save_test_result, test_source_hashes)
from .feature_cache import load_store
from .metric_utils import choose_device, seed_everything, write_json
from .utils import load_config, read_csv, resolve_path


def evaluate_final_test(config):
    """Load fixed best.pt and evaluate TEST once; this function never updates weights."""
    state = require_training_complete(config)
    receipt_path = resolve_path(config['paths']['final_test_receipt_json'])
    if receipt_path.exists() and read_json(receipt_path)['run_id'] == state['run_id']:
        raise ValueError('Final test already evaluated for this training run; use saved results')
    test_source_hashes(config)
    settings = config['training']
    seed_everything(config['seed'], settings['deterministic'], settings['num_threads'])
    device = choose_device(settings['device'])
    model, checkpoint = load_encoder(config['paths']['best_checkpoint'], config, device)
    if checkpoint.get('experiment_run_id') != state['run_id']:
        raise ValueError('Best checkpoint does not belong to this completed training run')
    candidate_store = load_store(config, 'test_candidate')
    query_store = load_store(config, 'test_query', allow_empty=True)
    predictions, _, _ = retrieve(model, candidate_store, query_store, device,
                                 config['inference']['batch_size'], config['inference']['top_k'])
    # Ground truth is opened only here, after training and checkpoint selection.
    truth = read_csv(config['paths']['test_ground_truth_csv'], ['question_id', 'player_id'])
    metrics, details = evaluate(predictions, truth)
    metadata = save_test_result(config, 'triplet', predictions, metrics, details, state)
    write_json({**metadata, 'status': 'completed', 'best_epoch': checkpoint['epoch'],
                'metrics': metrics}, receipt_path)
    print('FINAL HELD-OUT TEST RESULT', flush=True)
    for name, value in metrics.items():
        print(f'{name}: {value:.6f}', flush=True)
    return metrics


def main():
    """Evaluate a completed experiment using its audited test split."""
    args = experiment_parser(__doc__).parse_args()
    evaluate_final_test(load_config(args.config))


if __name__ == '__main__':
    main()
