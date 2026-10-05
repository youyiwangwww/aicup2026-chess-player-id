"""SGF parsing via sgfmill, following only the main variation."""
from sgfmill import sgf


def parse_target_moves(content, color, opening_moves=40, board_size=19):
    """Return target-color coordinates within the first N total moves.

    Pass counts as a move but has no coordinate. Setup stones do not count.
    sgfmill coordinates use row zero at the bottom and column zero at the left.
    """
    if opening_moves < 1:
        raise ValueError('opening_moves must be positive')
    color = str(color).strip().lower()
    if color not in ('b', 'w'):
        raise ValueError(f'Invalid target color: {color!r}')
    try:
        game = sgf.Sgf_game.from_string(content)
        if game.get_size() != board_size:
            raise ValueError(f'Expected board size {board_size}, got {game.get_size()}')
        moves, count = [], 0
        for node in game.get_main_sequence():
            move_color, point = node.get_move()
            if move_color is None:
                continue
            count += 1
            if move_color == color and point is not None:
                moves.append(point)
            if count >= opening_moves:
                break
        return moves
    except Exception as exc:
        raise ValueError(f'SGF parsing failed: {exc}') from exc
