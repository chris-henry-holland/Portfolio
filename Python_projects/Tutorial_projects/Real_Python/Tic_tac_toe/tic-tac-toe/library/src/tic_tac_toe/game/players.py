# tic_tac_toe/game/players.py

from __future__ import annotations

import abc
import time

from tic_tac_toe.logic.exceptions import InvalidMove
from tic_tac_toe.logic.minimax import findBestMove
from tic_tac_toe.logic.models import GameState, Mark, Move

class Player(metaclass=abc.ABCMeta):
    def __init__(self, mark: Mark) -> None:
        self.mark = mark

    def makeMove(self, game_state: GameState) -> GameState:
        if self.mark is not game_state.current_mark:
            raise InvalidMove("It is the other player's turn")
        if move := self.getMove(game_state):
            return move.after_state
        raise InvalidMove("No more possible moves")
        return
    
    @abc.abstractmethod
    def getMove(self, game_state: GameState) -> Move | None:
        """
        Returns the current player's move in the given game state.
        """

class ComputerPlayer(Player, metaclass=abc.ABCMeta):
    def __init__(self, mark: Mark, move_delay_s: float=0.25, **kwargs) -> None:
        super().__init__(mark)
        self.move_delay_s = move_delay_s
    
    def getMove(self, game_state: GameState) -> Move | None:
        time.sleep(self.move_delay_s)
        return self.getComputerMove(game_state)

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

class MinimaxComputerPlayer(ComputerPlayer):

    def getComputerMove(self, game_state: GameState) -> Move | None:
        if game_state.game_not_started:
            return game_state.makeRandomMove()
        return findBestMove(game_state)