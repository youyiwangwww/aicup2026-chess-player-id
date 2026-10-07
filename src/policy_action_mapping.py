"""CPU-only 19x19 SGF / MiniZero bottom-left-origin action mapping."""
from numbers import Integral

BOARD_SIZE=19
PASS_INDEX=BOARD_SIZE*BOARD_SIZE


def sgf_to_action(coordinate):
    """Map SGF top-left-origin coordinates to MiniZero index; empty/None means pass."""
    if coordinate is None or coordinate=='':
        return PASS_INDEX
    if not isinstance(coordinate,str) or len(coordinate)!=2 or any(c not in 'abcdefghijklmnopqrs' for c in coordinate):
        raise ValueError('Expected lowercase 19x19 SGF coordinate or empty pass')
    x,y=(ord(c)-ord('a') for c in coordinate)
    return (BOARD_SIZE-1-y)*BOARD_SIZE+x


def action_to_sgf(action):
    """Invert action index 0..361, preserving pass as the empty SGF value."""
    if isinstance(action,bool) or not isinstance(action,Integral) or not 0<=action<=PASS_INDEX:
        raise ValueError('Expected integer action index 0..361')
    action=int(action)
    if action==PASS_INDEX:
        return ''
    return chr(ord('a')+action%BOARD_SIZE)+chr(ord('a')+BOARD_SIZE-1-action//BOARD_SIZE)
