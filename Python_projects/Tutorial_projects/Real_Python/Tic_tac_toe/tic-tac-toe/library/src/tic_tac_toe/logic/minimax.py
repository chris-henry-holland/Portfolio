# tic_tac_toe/logic/minimax.py

from __future__ import annotations

from functools import partial

from tic_tac_toe.logic.models import GameState, Mark, Move

def minimax(
    move: Move,
    maximiser: Mark,
    choose_highest_score: bool=False
) -> int:
    if move.after_state.game_over:
        return move.after_state.evaluateScore(maximiser)
    func = max if choose_highest_score else min
    return func(
        minimax(next_move, maximiser, not choose_highest_score)
        for next_move in move.after_state.possible_moves
    )

def findBestMove(game_state: GameState) -> Move | None:
    maximiser: Mark = game_state.current_mark
    bound_minimax = partial(minimax, maximiser=maximiser)
    return max(game_state.possible_moves, key=bound_minimax)