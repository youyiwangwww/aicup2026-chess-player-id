"""Target-turn windows and game-count-weighted color-aware opening scores."""
import numpy as np
import pandas as pd
from sgfmill import sgf
from sklearn.metrics.pairwise import cosine_similarity

from .sgf_parser import parse_target_moves

METHODS = [('Opening-total-10', 'total', 10, False), ('Opening-player-5', 'player', 5, False),
           ('Opening-player-10', 'player', 10, False), ('Opening-player-20', 'player', 20, False),
           ('Color-aware-player-5', 'player', 5, True), ('Color-aware-player-10', 'player', 10, True)]


def player_moves(content, color, count, board_size=19):
    """Count target turns including passes; setup and opponent turns do not consume budget."""
    color = str(color).strip().lower()
    if color not in ('b', 'w') or count < 1:
        raise ValueError('Invalid target color or player move budget')
    try:
        game = sgf.Sgf_game.from_string(content)
        if game.get_size() != board_size:
            raise ValueError('Expected 19x19')
        points, used = [], 0
        for node in game.get_main_sequence():
            c, point = node.get_move()
            if c != color:
                continue
            used += 1
            if point is not None:
                points.append(point)
            if used == count:
                break
        return points
    except Exception as exc:
        raise ValueError(f'SGF parsing failed: {exc}') from exc


def fingerprints(frame, group_column, mode, count):
    """Return mixed and separate B/W fingerprints; record per-game failures."""
    values, errors = {}, []
    for row in frame.itertuples(index=False):
        identity = getattr(row, group_column)
        color = row.color.upper()
        try:
            points = (parse_target_moves(row.sgf_content, color, count) if mode == 'total'
                      else player_moves(row.sgf_content, color, count))
            heat = np.zeros((19, 19), dtype=np.float64)
            for point in points:
                heat[point] += 1
            if not heat.any():
                raise ValueError('No non-pass target move in window')
            values.setdefault((identity, color), []).append(heat.ravel())
        except ValueError as exc:
            errors.append({'group_id': identity, 'game_id': row.game_id, 'error': str(exc)})
    separate = {key: np.mean(items, axis=0) for key, items in values.items()}
    mixed = {}
    for identity in frame[group_column].unique():
        items = [value for c in ['B', 'W'] for value in values.get((identity, c), [])]
        mixed[identity] = np.mean(items, axis=0) if items else np.zeros(361)
    return mixed, separate, errors


def opening_scores(candidates, queries, mode, count, color_aware=False):
    """Match only like colors when requested; retain all candidates and original game weights."""
    candidates, queries = candidates.copy(), queries.copy()
    candidates['color'] = candidates.color.str.strip().str.upper()
    queries['color'] = queries.color.str.strip().str.upper()
    players, questions = sorted(candidates.player_id.unique()), sorted(queries.question_id.unique())
    cm, cs, ce = fingerprints(candidates, 'player_id', mode, count)
    qm, qs, qe = fingerprints(queries, 'question_id', mode, count)
    if not color_aware:
        scores = cosine_similarity(np.stack([qm[q] for q in questions]), np.stack([cm[p] for p in players]))
    else:
        scores = np.zeros((len(questions), len(players)))
        counts = queries.groupby(['question_id', 'color']).size()
        totals = queries.groupby('question_id').size()
        for color in ['B', 'W']:
            a = np.stack([qs.get((q, color), np.zeros(361)) for q in questions])
            b = np.stack([cs.get((p, color), np.zeros(361)) for p in players])
            weights = np.array([counts.get((q, color), 0) / totals[q] for q in questions])
            scores += cosine_similarity(a, b) * weights[:, None]
    missing = sum((p, c) not in cs for p in players for c in ['B', 'W'])
    return scores, players, questions, ce + qe, missing
