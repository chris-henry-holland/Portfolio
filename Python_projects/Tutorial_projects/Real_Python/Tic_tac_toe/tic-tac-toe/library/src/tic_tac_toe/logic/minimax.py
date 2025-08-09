# tic_tac_toe/logic/minimax.py

from __future__ import annotations

import json
import os
import random

from collections import deque
from functools import cache, partial
from importlib import resources

from tic_tac_toe.logic.models import Grid, GameState, Mark, Move

class MinimaxSerialiser:
    DEFAULT_FILENAME = "minimax.json"

    @staticmethod
    def serialiseKey(game_state: GameState) -> str:
        return str(game_state.encoding)

    @staticmethod
    def serialiseValue(game_state_score: int, game_state_optimal_moves: list[Move]) -> str:
        lst = [
            ",".join(str(gs.encoding) for gs in game_state_optimal_moves),
            str(game_state_score),
        ]
        return ";".join(lst)
        #for game_state2 in game_state_optimal_moves:
        #    lst.append(game_state2)
        #return "".join([move.before_state.grid.cells, move.after_state.grid.cells])

    @staticmethod
    def deserialiseKey(serialised_key: str) -> GameState:
        return GameState.decode(int(serialised_key))
    
    @staticmethod
    def deserialiseValue(serialised_value: str) -> tuple[int, tuple[Move]]:
        optimal_moves_str, score_str = serialised_value.split(";")
        optimal_moves = tuple(Move.decode(int(x)) for x in optimal_moves_str.split(",")) if optimal_moves_str else ()
        return (int(score_str), optimal_moves)

    @staticmethod
    @cache
    def load(filename: str=DEFAULT_FILENAME, create_if_not_exist: bool=True) -> dict[GameState, tuple[Move, tuple[Move]]]:
        print("Using MinimaxSerialiser.load()")
        #print(filename)
        #print(os.path.join(__package__, filename))
        print(__name__, __package__, __file__)
        #if not os.path.isfile(os.path.join(__package__, filename)):
        if not resources.is_resource(__package__, filename):
            print("file does not yet exist")
            if not create_if_not_exist:
                raise FileNotFoundError("The specified file name does not exist")
            MinimaxSerialiser.dump(filename=filename)
        else: print("file exists")
        
        with resources.open_text(__package__, filename, encoding="utf-8") as file:
            str_dict = json.load(file)
        return {
            MinimaxSerialiser.deserialiseKey(k): MinimaxSerialiser.deserialiseValue(v)
            for k, v in str_dict.items()
        }
    
    @staticmethod
    def dump(filename: str=DEFAULT_FILENAME) -> None:
        print("Using MinimaxSerialiser.dump()")
        path_lst = __file__.split("/")
        print(path_lst)
        path_lst.pop()
        print(os.path.isdir(os.path.join(*path_lst)))
        path_lst.append(filename)
        file_abs = "/".join(path_lst)
        print(f"file_abs = {file_abs}")
        with open(file_abs, mode="w", encoding="utf-8") as file:
            print(file)
            score_dict = MinimaxSerialiser.precomputeScores()
            print(score_dict)
            json.dump(score_dict, file)
        return
    
    @staticmethod
    def precomputeScores() -> dict[str, list[int]]:
        minimax_dict = minimaxFull(GameState(Grid(), Mark("X")), maximiser=Mark("X"))
        return {
            MinimaxSerialiser.serialiseKey(k): MinimaxSerialiser.serialiseValue(*v)
            for k, v in minimax_dict.items()
        }

def minimaxFull(
    start_game_state: GameState,
    maximiser: Mark,
) -> dict[GameState, tuple[int, tuple[GameState]]]:

    memo = {}
    def recur(curr_gamestate: GameState) -> tuple[int, tuple[Move]]:
        args = curr_gamestate
        if args in memo.keys(): return memo[args]
        if curr_gamestate.game_over:
            res = (curr_gamestate.evaluateScore(maximiser), ())
            memo[args] = res
            return res
        neg = (curr_gamestate.current_mark == Mark("O"))
        func = (lambda a, b: a > b) if neg else (lambda a, b: a < b)
        res = [-float("inf"), []] if neg else [float("inf"), []]
        for move in curr_gamestate.possible_moves:
            nxt_gamestate = move.after_state
            ans = recur(nxt_gamestate)
            if ans[0] == res[0]:
                res[1].append(move)
                continue
            if func(ans[0], res[0]):
                res = [ans[0], [move]]
        res = (res[0], tuple(res[1]))
        memo[args] = res
        return res
    recur(start_game_state)
    return memo

def minimax(
    game_state: GameState,
    maximiser: Mark,
) -> tuple[int, tuple[GameState]]:
    res = minimaxFull(game_state, maximiser)
    return res[game_state]

def findBestMove(game_state: GameState) -> Move | None:
    maximiser: Mark = game_state.current_mark
    max_score, max_score_moves = minimax(game_state, maximiser)
    return random.choice(max_score_moves)

def findBestMovePrecomputed(game_state: GameState, filename: str=MinimaxSerialiser.DEFAULT_FILENAME) -> Move | None:
    moves_dict = MinimaxSerialiser.load(filename=filename, create_if_not_exist=True)
    if game_state in moves_dict.keys():
        max_score, max_score_moves = moves_dict[game_state]
        return random.choice(max_score_moves)
    game_state_inv = game_state.inverted_game_state
    max_score, max_score_moves = moves_dict[game_state_inv]
    return random.choice(max_score_moves).inverted_move