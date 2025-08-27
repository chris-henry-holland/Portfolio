# tic_tac_toe/game/players.py

from __future__ import annotations

from typing import Callable

import abc
import enum
import time

from tic_tac_toe.logic.exceptions import InvalidMove
from tic_tac_toe.logic.minimax import findBestMove, findBestMovePrecomputed
from tic_tac_toe.logic.models import Grid, GameState, Mark, Move

class PlayerType(enum.StrEnum):
    HUMAN = "human"
    RANDOM = "random"
    MINIMAX = "minimax"

    @property
    def is_computer(self) -> bool:
        return self != PlayerType.HUMAN

class Player(metaclass=abc.ABCMeta):
    def __init__(self, mark: Mark) -> None:
        self.mark = mark

    def makeMove(self, game_state: GameState | int) -> GameState | None:
        if self.mark is not game_state.current_mark:
            raise InvalidMove("It is the other player's turn")
        if move := self.getMove(game_state):
            if move == -1:
                return None
            return move.after_state
        raise InvalidMove("No more possible moves")
    
    @abc.abstractmethod
    def getMove(self, game_state: GameState) -> Move | None:
        """
        Returns the current player's move in the given game state.
        """

    #def resetRequest(self) -> 
class ComputerPlayer(Player, metaclass=abc.ABCMeta):
    def __init__(self, mark: Mark, move_delay_s: float=0.25, **kwargs) -> None:
        super().__init__(mark)
        self.move_delay_s = move_delay_s
    
    def getMove(self, game_state: GameState) -> Move | None:
        since = time.time()
        res = self.getComputerMove(game_state)
        intvl = time.time() - since
        time.sleep(max(0, self.move_delay_s - intvl))
        return res

    @abc.abstractmethod
    def getComputerMove(self, game_state: GameState) -> Move | None:
        """
        Returns the computer's move for the given game state.
        """

class RandomComputerPlayer(ComputerPlayer):
    def getComputerMove(self, game_state: GameState) -> Move | None:
        return game_state.makeRandomMove()
        #try:
        #    return random.choice(game_state.possible_moves)
        #except IndexError:
        #    return None

class MinimaxComputerPlayerV1(ComputerPlayer):
    def getComputerMove(self, game_state: GameState) -> Move | None:
        if game_state.game_not_started:
            return game_state.makeRandomMove()
        return findBestMove(game_state)

class MinimaxComputerPlayerV2(ComputerPlayer):
    def getComputerMove(self, game_state: GameState) -> Move | None:
        if game_state.game_not_started:
            return game_state.makeRandomMove()
        #findBestMove(game_state)
        return findBestMovePrecomputed(game_state)

MinimaxComputerPlayer = MinimaxComputerPlayerV2