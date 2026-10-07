"""Small AI CUP row adapter; no files, bindings, policy network or batch dataset export."""
from dataclasses import dataclass
from typing import Mapping,Optional
from sgfmill import sgf
from .policy_action_mapping import sgf_to_action


@dataclass(frozen=True)
class PlayerMove:
    """Actual recorded move and target-side numbering; pass counts as a move."""
    color:str
    coordinate:str
    action_index:int
    total_move_number:int
    target_move_number:Optional[int]


@dataclass(frozen=True)
class MiniZeroRecord:
    """Keep target identity metadata outside model-input SGF to avoid label leakage."""
    game_id:str
    target_color:str
    board_size:int
    sgf_content:str
    moves:tuple
    komi:Optional[float]=None
    player_id:Optional[str]=None
    rank:Optional[str]=None

    @property
    def target_moves(self):
        """Select the actual target player's moves using CSV color, never alternation guesses."""
        return tuple(m for m in self.moves if m.color==self.target_color)


def _serialize(moves,komi=None):
    """Emit one-line MiniZero SGF dialect with explicit B/W order and canonical empty pass."""
    game=sgf.Sgf_game(19)
    root=game.get_root()
    root.set_raw('GM',b'go_19x19')
    root.set('CA','UTF-8')
    if komi is not None:
        root.set('KM',komi)
    for move in moves:
        child=game.extend_main_sequence()
        child.set_raw(move.color,move.coordinate.encode('ascii'))
    return game.serialise(wrap=None).decode('utf-8').strip()


def adapt_aicup_row(row:Mapping):
    """Parse only the main line; reject setup/turn overrides rather than silently losing them."""
    try:
        if not isinstance(row['sgf_content'],str) or not row['sgf_content'].strip():
            raise ValueError('Missing SGF content')
        color=str(row['color']).strip().upper()
        if color not in ('B','W'):
            raise ValueError('CSV color must be B/W')
        game=sgf.Sgf_game.from_string(row['sgf_content'])
        if game.get_size()!=19:
            raise ValueError('Policy v1 requires 19x19; board size will not be rewritten')
        moves=[]
        target_number=0
        for node in game.get_main_sequence():
            if any(node.has_property(p) for p in ['AB','AW','AE','PL','HA']):
                raise ValueError('Setup/handicap/turn override requires a separately verified environment adapter')
            if node.has_property('B') and node.has_property('W'):
                raise ValueError('Ambiguous node with both B/W')
            move_color,point=node.get_move()
            if move_color is None:
                continue
            move_color=move_color.upper()
            coordinate='' if point is None else chr(ord('a')+point[1])+chr(ord('a')+18-point[0])
            if move_color==color:
                target_number+=1
            moves.append(PlayerMove(move_color,coordinate,sgf_to_action(coordinate),len(moves)+1,
                                    target_number if move_color==color else None))
        komi=game.get_root().get('KM') if game.get_root().has_property('KM') else None
        return MiniZeroRecord(str(row['game_id']),color,19,_serialize(moves,komi),tuple(moves),komi,
                              str(row['player_id']) if row.get('player_id') is not None else None,
                              str(row['rank']) if row.get('rank') is not None else None)
    except Exception as exc:
        raise ValueError(f'AI CUP SGF adapter failed: {exc}') from exc


def sgf_before_move(record:MiniZeroRecord,total_move_number:int):
    """Create a tiny prefix BEFORE the actual move; first move has an empty-board prefix."""
    if isinstance(total_move_number,bool) or not isinstance(total_move_number,int) or not 1<=total_move_number<=len(record.moves):
        raise ValueError('Move number outside recorded main line')
    return _serialize(record.moves[:total_move_number-1],record.komi)
