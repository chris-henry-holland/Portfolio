# frontends/console/players.py

from __future__ import annotations

import re

from tic_tac_toe.game.players import Player
from tic_tac_toe.logic.exceptions import InvalidMove
from tic_tac_toe.logic.models import GameState, Move

class ConsolePlayer(Player):

    def getMove(self, game_state: GameState) -> Move | None:
        while not game_state.game_over:
            try:
                index = gridToIndex(input(f"{self.mark}'s move: ").strip())
            except ValueError:
                print("Please provide coordinates in the form of A1 or 1A")
            else:
                try:
                    return game_state.makeMoveTo(index)
                except InvalidMove:
                    print("That cell is already occupied.")
        return None

ord_A = ord("A")

def gridToIndex(grid: str) -> int:
    if re.match(r"[abcABC][123]", grid):
        col, row = grid
    elif re.match(r"[123][abcABC]", grid):
        row, col = grid
    else: raise ValueError("Invalid grid coordinates")
    return 3 * (int(row) - 1) + (ord(col.upper()) - ord_A)