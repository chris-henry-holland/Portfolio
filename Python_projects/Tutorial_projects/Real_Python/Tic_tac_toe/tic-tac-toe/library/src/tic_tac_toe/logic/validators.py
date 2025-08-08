# tic_tac_toe/logic/validators.py

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tic_tac_toe.game.players import Player
    from tic_tac_toe.logic.models import GameState, Grid, Mark

import re

from tic_tac_toe.logic.exceptions import InvalidGameState

def validateGrid(grid: Grid) -> None:
    if not re.match(r"^[\sXO]{9}", grid.cells):
        raise ValueError("cells must contain exactly 9 cells of: X, O or space")
    return

def validateGameState(game_state: GameState) -> None:
    validateNumberOfMarks(game_state.grid)
    validateStartingMark(game_state.grid, game_state.starting_mark)
    validateWinner(
        game_state.grid, game_state.starting_mark, game_state.winner,
    )
    return

def validateNumberOfMarks(grid: Grid) -> None:
    if abs(grid.x_count - grid.o_count) > 1:
        raise InvalidGameState("Wrong number of Xs and Os")
    return

def validateStartingMark(grid: Grid, starting_mark: Mark) -> None:
    if grid.x_count == grid.o_count:
        return
    majority_mark = "X" if grid.x_count > grid.o_count else "O"
    if starting_mark != majority_mark:
        raise InvalidGameState("Wrong starting mark")
    return

def validateWinner(
    grid: Grid,
    starting_mark: Mark,
    winner: Mark | None,
) -> None:
    if winner is None: return
    winner_count_attr = f"{winner.lower()}_count"
    other_count_attr = f"{winner.other.lower()}_count"
    if getattr(grid, winner_count_attr) != getattr(grid, other_count_attr) + (starting_mark == winner):
        raise InvalidGameState(f"Wrong number of {winner}s")
    return

def validatePlayers(player1: Player, player2: Player) -> None:
    if player1.mark is player2.mark:
        raise ValueError("Players must use different marks")