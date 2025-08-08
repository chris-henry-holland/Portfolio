# tic_tac_toe/game/engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeAlias

from tic_tac_toe.game.players import Player
from tic_tac_toe.game.renderers import Renderer
from tic_tac_toe.logic.exceptions import InvalidMove
from tic_tac_toe.logic.models import GameState, Grid, Mark
from tic_tac_toe.logic.validators import validatePlayers

ErrorHandler: TypeAlias = Callable[[Exception], None]

@dataclass(frozen=True)
class TicTacToe:
    player1: Player
    player2: Player
    renderer: Renderer
    error_handler: ErrorHandler | None = None

    def __post_init__(self):
        validatePlayers(self.player1, self.player2)

    def play(self, starting_mark: Mark = Mark("X")) -> None:
        game_state = GameState(Grid(), starting_mark)
        while True:
            self.renderer.render(game_state)
            if game_state.game_over:
                break
            player = self.getCurrentPlayer(game_state)
            try:
                game_state = player.makeMove(game_state)
            except InvalidMove as  ex:
                if self.error_handler:
                    self.error_handler(ex)
        return
    
    def getCurrentPlayer(self, game_state: GameState) -> Player:
        return self.player1 if game_state.current_mark is self.player1.mark else self.player2
        