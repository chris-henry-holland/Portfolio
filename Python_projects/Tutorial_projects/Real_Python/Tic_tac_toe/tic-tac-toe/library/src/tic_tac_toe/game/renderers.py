# tic_tac_toe/game/renderers.py

from __future__ import annotations

import abc

from tic_tac_toe.logic.models import GameState

class Renderer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def render(self, game_state: GameState) -> None:
        """
        Renders the given game state.
        """