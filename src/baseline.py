"""Rank candidates using cosine similarity between opening fingerprints."""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .opening_features import build_fingerprints
from .utils import argument_parser, load_config, read_csv, write_csv


def predict(candidate_fingerprints, query_fingerprints, top_k=5):
    """Return long-form rankings with deterministic player-ID tie breaking."""
    if not 1 <= top_k <= 5:
        raise ValueError('top_k must be between 1 and 5')
    if not candidate_fingerprints:
        raise ValueError('No valid candidate fingerprints; inspect parsing errors')
    players = sorted(candidate_fingerprints)
    matrix = np.stack([candidate_fingerprints[p] for p in players])
    rows = []
    for question, fingerprint in sorted(query_fingerprints.items()):
        similarities = cosine_similarity(fingerprint.reshape(1, -1), matrix)[0]
        order = np.argsort(-similarities, kind='stable')[:top_k]
        for rank, index in enumerate(order, 1):
            rows.append({'question_id': question, 'rank': rank,
                         'player_id': players[index], 'similarity': similarities[index]})
    return pd.DataFrame(rows, columns=['question_id', 'rank', 'player_id', 'similarity'])


def main():
    """Build fingerprints, record errors and save local Top-5 predictions."""
    args = argument_parser(__doc__).parse_args()
    config = load_config(args.config)
    paths = config['paths']
    candidates = read_csv(paths['candidates_csv'], ['player_id', 'game_id', 'color', 'sgf_content'])
    queries = read_csv(paths['queries_csv'], ['question_id', 'game_id', 'color', 'sgf_content'])
    cfp, cerr = build_fingerprints(candidates, 'player_id', **config['features'])
    qfp, qerr = build_fingerprints(queries, 'question_id', **config['features'])
    errors = pd.concat([cerr, qerr], ignore_index=True)
    write_csv(errors, paths['parsing_errors_csv'])
    result = predict(cfp, qfp, **config['baseline'])
    write_csv(result, paths['predictions_csv'])
    print(f'Valid candidates: {len(cfp)}/{candidates.player_id.nunique()}; '
          f'valid questions: {len(qfp)}/{queries.question_id.nunique()}; skipped games: {len(errors)}')
    print('Questions without valid games receive zero during evaluation.')


if __name__ == '__main__':
    main()
