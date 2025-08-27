# frontends/window/players.py

from __future__ import annotations

from queue import Queue

from tic_tac_toe.game.players import Player
from tic_tac_toe.logic.models import GameState, Mark, Move

class WindowPlayer(Player):
    def __init__(self, mark: Mark, events: Queue, **kwargs) -> None:
        super().__init__(mark)
        self.events = events
    
    def getMove(self, game_state: GameState) -> Move | int | None:
        idx = self.events.get()
        if idx == -1:
            return -1
        try:
            return game_state.makeMoveTo(idx)
        finally:
            self.events.task_done()