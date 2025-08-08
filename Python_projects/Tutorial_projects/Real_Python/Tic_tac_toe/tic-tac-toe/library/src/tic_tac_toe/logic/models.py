# tic_tac_toe/logic/models.py

from __future__ import annotations

import enum
import random
import re

from dataclasses import dataclass
from functools import cached_property

from tic_tac_toe.logic.exceptions import InvalidMove, UnknownGameScore
from tic_tac_toe.logic.validators import validateGameState, validateGrid


WINNING_PATTERNS = (
    "???......",
    "...???...",
    "......???",
    "?..?..?..",
    ".?..?..?.",
    "..?..?..?",
    "?...?...?",
    "..?.?.?..",
)

class Mark(enum.StrEnum):
    CROSS = "X"
    NAUGHT = "O"

    @property
    def other(self) -> Mark:
        return Mark.CROSS if self is Mark.NAUGHT else Mark.NAUGHT

@dataclass(frozen=True)
class Grid:
    cells: str = " " * 9

    def __post_init__(self) -> None:
        validateGrid(self)
        #if not re.match(r"^[\sXO]{9}", self.cells):
        #    raise ValueError("cells must contain exactly 9 cells of: X, O or space")
    
    @cached_property
    def x_count(self) -> int:
        return self.cells.count("X")
    
    @cached_property
    def o_count(self) -> int:
        return self.cells.count("O")
    
    @cached_property
    def empty_count(self) -> int:
        return self.cells.count(" ")

@dataclass(frozen=True)
class Move:
    mark: Mark
    cell_idx: int
    before_state: GameState
    after_state: GameState

@dataclass(frozen=True)
class GameState:
    grid: Grid
    starting_mark: Mark = Mark("X")

    def __post_init__(self) -> None:
        validateGameState(self)

    @cached_property
    def current_mark(self) -> Mark:
        if self.grid.x_count == self.grid.o_count:
            return self.starting_mark
        else:
            return self.starting_mark.other
    
    @cached_property
    def game_not_started(self) -> bool:
        return self.grid.empty_count == 9
    
    @cached_property
    def game_over(self) -> bool:
        return self.winner is not None or self.tie
    
    @cached_property
    def tie(self) -> bool:
        return self.winner is None and self.grid.empty_count == 0
    
    @cached_property
    def winner(self) -> Mark | None:
        for pattern in WINNING_PATTERNS:
            for mark in Mark:
                if re.match(pattern.replace("?", mark), self.grid.cells):
                    return mark
        return None

    @cached_property
    def winning_cells(self) -> list[int]:
        for pattern in WINNING_PATTERNS:
            for mark in Mark:
                if re.match(pattern.replace("?", mark), self.grid.cells):
                    return [
                        match.start()
                        for match in re.finditer(r"\?", pattern)
                    ]
        return []
    
    @cached_property
    def possible_moves(self) -> list[Move]:
        if self.game_over:
            return []
        moves = [
            self.makeMoveTo(match.start())
            for match in re.finditer(r"\s", self.grid.cells)
        ]
        return moves

    def makeMoveTo(self, idx: int) -> Move:
        if self.grid.cells[idx] != " ":
            raise InvalidMove("Cell is not empty")
        return Move(
            mark=self.current_mark,
            cell_idx=idx,
            before_state=self,
            after_state=GameState(
                Grid(
                    "".join([
                        self.grid.cells[:idx],
                        self.current_mark,
                        self.grid.cells[idx + 1:],
                    ]),
                ),
                self.starting_mark,
            ),
        )
    
    def makeRandomMove(self) -> Move | None:
        try:
            return random.choice(self.possible_moves)
        except IndexError:
            return None

    def evaluateScore(self, mark: Mark) -> int:
        if not self.game_over:
            raise UnknownGameScore("Game is not over yet")
        if self.tie: return 0
        return 1 if self.winner is mark else -1