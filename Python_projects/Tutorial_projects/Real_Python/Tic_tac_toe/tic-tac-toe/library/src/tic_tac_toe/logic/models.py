# tic_tac_toe/logic/models.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tic_tac_toe.logic.models import Grid

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
    orig: Grid | None = None

    def __post_init__(self) -> None:
        validateGrid(self)
        #if not re.match(r"^[\sXO]{9}", self.cells):
        #    raise ValueError("cells must contain exactly 9 cells of: X, O or space")
    
    @cached_property
    def encoding(self) -> int:
        res = 0
        for l in reversed(self.cells):
            d = 0
            if l == "X": d = 1
            elif l == "O": d = 2
            res = res * 3 + d
        return res
    
    @staticmethod
    def decode(num: int) -> Grid:
        cells = []
        for _ in range(9):
            num, d = divmod(num, 3)
            l = " "
            if d == 1: l = "X"
            elif d == 2: l = "O"
            cells.append(l)
        return Grid("".join(cells))
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Grid): return False
        return self.encoding == other.encoding

    def __hash__(self) -> int:
        return self.hsh

    @cached_property
    def hsh(self) -> int:
        return hash(self.encoding)#hash(self.cells)

    @cached_property
    def x_count(self) -> int:
        return self.cells.count("X")
    
    @cached_property
    def o_count(self) -> int:
        return self.cells.count("O")
    
    @cached_property
    def empty_count(self) -> int:
        return self.cells.count(" ")
    
    @cached_property
    def inverted_grid(self) -> Grid:
        if self.orig is not None:
            return self.orig
        res = []
        for l in self.cells:
            if l == "O": res.append("X")
            elif l == "X": res.append("O")
            else: res.append(l)
        return Grid("".join(res), orig=self)

@dataclass(frozen=True)
class Move:
    mark: Mark
    cell_idx: int
    before_state: GameState
    after_state: GameState
    orig: Move | None = None

    @cached_property
    def inverted_move(self) -> Move:
        if self.orig is not None:
            return self.orig
        return Move(
            self.mark.other,
            self.cell_idx,
            self.before_state.inverted_game_state,
            self.after_state.inverted_game_state,
            orig=self,
        )

    @cached_property
    def encoding(self) -> int:
        res = self.before_state.encoding * (3 ** 9 * 2) + self.after_state.encoding
        res = res * 9 + self.cell_idx
        res = (res << 1) + (self.mark == Mark("X"))
        return res
    
    @staticmethod
    def decode(num: int) -> Move:
        mark = Mark("X") if num & 1 else Mark("O")
        num >>= 1
        num, cell_idx = divmod(num, 9)
        before_enc, after_enc = divmod(num, 3 ** 9 * 2)
        before_state = GameState.decode(before_enc)
        after_state = GameState.decode(after_enc)
        return Move(mark, cell_idx, before_state, after_state)
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Move): return False
        return self.encoding == other.encoding

@dataclass(frozen=True)
class GameState:
    grid: Grid
    starting_mark: Mark = Mark("X")
    orig: GameState | None = None

    def __post_init__(self) -> None:
        validateGameState(self)
    
    @cached_property
    def encoding(self) -> int:
        res = self.grid.encoding << 1
        if self.starting_mark == Mark("X"):
            res += 1
        return res
    
    @staticmethod
    def decode(num: int) -> Grid:
        starting_mark = Mark("X" if num & 1 else "O")
        grid = Grid.decode(num >> 1)
        return GameState(grid, starting_mark)
        """
        cells = []
        for _ in range(9):
            num, d = divmod(num, 3)
            l = " "
            if d == 1: l = "X"
            elif d == 2: l = "O"
            cells.append(d)
        return Grid("".join(cells))
        """

    def __hash__(self) -> int:
        return self.hsh

    @cached_property
    def hsh(self) -> int:
        return hash(self.encoding)
        #return hash((self.grid, self.starting_mark))
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, GameState): return False
        return self.encoding == other.encoding
    
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
    
    @cached_property
    def inverted_game_state(self) -> GameState:
        if self.orig is not None:
            return self.orig
        return GameState(
            self.grid.inverted_grid,
            starting_mark=self.starting_mark.other,
            orig=self,
        )

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