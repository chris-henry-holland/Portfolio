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
    def serialiseGameState(game_state: GameState) -> str:
        return str(game_state.encoding)

    @staticmethod
    def serialiseOptimalMoves(game_state_score: int, game_state_optimal_moves: list[Move]) -> str:
        lst = [
            ",".join(str(gs.encoding) for gs in game_state_optimal_moves),
            str(game_state_score),
        ]
        return ";".join(lst)
        #for game_state2 in game_state_optimal_moves:
        #    lst.append(game_state2)
        #return "".join([move.before_state.grid.cells, move.after_state.grid.cells])

    @staticmethod
    def deserialiseGameState(serialised_key: str) -> GameState:
        return GameState.decode(int(serialised_key))
    
    @staticmethod
    def deserialiseOptimalMoves(serialised_value: str) -> tuple[int, tuple[Move]]:
        optimal_moves_str, score_str = serialised_value.split(";")
        optimal_moves = tuple(Move.decode(int(x)) for x in optimal_moves_str.split(",")) if optimal_moves_str else ()
        return (int(score_str), optimal_moves)

    @staticmethod
    @cache
    def loadSerialised(filename: str=DEFAULT_FILENAME, create_if_not_exist: bool=True) -> dict[str, str]:
        if not resources.is_resource(__package__, filename):
            #print("file does not yet exist")
            if not create_if_not_exist:
                raise FileNotFoundError("The specified file name does not exist")
            MinimaxSerialiser.dump(filename=filename)
        #else: print("file exists")
        
        with resources.open_text(__package__, filename, encoding="utf-8") as file:
            str_dict = json.load(file)
        return str_dict

    @staticmethod
    @cache
    def load(filename: str=DEFAULT_FILENAME, create_if_not_exist: bool=True) -> dict[GameState, tuple[Move, tuple[Move]]]:
        """
        #print("Using MinimaxSerialiser.load()")
        #print(filename)
        #print(os.path.join(__package__, filename))
        #print(__name__, __package__, __file__)
        #if not os.path.isfile(os.path.join(__package__, filename)):
        if not resources.is_resource(__package__, filename):
            #print("file does not yet exist")
            if not create_if_not_exist:
                raise FileNotFoundError("The specified file name does not exist")
            MinimaxSerialiser.dump(filename=filename)
        #else: print("file exists")
        
        with resources.open_text(__package__, filename, encoding="utf-8") as file:
            str_dict = json.load(file)
        """
        str_dict = MinimaxSerialiser.loadSerialised(filename=filename, create_if_not_exist=create_if_not_exist)
        return {
            MinimaxSerialiser.deserialiseGameState(k): MinimaxSerialiser.deserialiseOptimalMoves(v)
            for k, v in str_dict.items()
        }

    @staticmethod
    @cache
    def loadGameStateOptimalMovesFromFile(game_state: GameState, filename: str=DEFAULT_FILENAME, create_if_not_exist: bool=True) -> tuple[Move, tuple[Move]] | None:
        serialised_dict = MinimaxSerialiser.loadSerialised(filename=filename, create_if_not_exist=create_if_not_exist)
        game_state_serialised = MinimaxSerialiser.serialiseGameState(game_state)
        res = serialised_dict.get(game_state_serialised, None)
        return None if res is None else MinimaxSerialiser.deserialiseOptimalMoves(res)

    @staticmethod
    def dump(filename: str=DEFAULT_FILENAME) -> None:
        #print("Using MinimaxSerialiser.dump()")
        path_lst = __file__.split("/")
        #print(path_lst)
        path_lst.pop()
        #print(os.path.isdir(os.path.join(*path_lst)))
        path_lst.append(filename)
        file_abs = "/".join(path_lst)
        #print(f"file_abs = {file_abs}")
        with open(file_abs, mode="w", encoding="utf-8") as file:
            #print(file)
            score_dict = MinimaxSerialiser.precomputeScores()
            #print(score_dict)
            json.dump(score_dict, file)
        return
    
    @staticmethod
    def precomputeScores() -> dict[str, list[int]]:
        minimax_dict = minimaxFull(GameState(Grid(), Mark("X")))#, maximiser=Mark("X"))
        return {
            MinimaxSerialiser.serialiseGameState(k): MinimaxSerialiser.serialiseOptimalMoves(*v)
            for k, v in minimax_dict.items()
        }

def minimaxFull(
    start_game_state: GameState,
) -> dict[GameState, tuple[int, tuple[GameState]]]:

    memo = {}
    def recur(curr_gamestate: GameState) -> tuple[int, tuple[Move]]:
        args = curr_gamestate
        if args in memo.keys(): return memo[args]
        if curr_gamestate.game_over:
            res = (curr_gamestate.evaluateScore(curr_gamestate.current_mark), ())
            memo[args] = res
            return res
        #neg = (curr_gamestate.current_mark == Mark("O"))
        #func = (lambda a, b: a < b)# if neg else (lambda a, b: a < b)
        res = [-float("inf"), []]# if neg else [float("inf"), []]
        for move in curr_gamestate.possible_moves:
            nxt_gamestate = move.after_state
            ans = recur(nxt_gamestate)
            if -ans[0] == res[0]:
                res[1].append(move)
                continue
            if -ans[0] > res[0]:
                res = [-ans[0], [move]]
        res = (res[0], tuple(res[1]))
        memo[args] = res
        return res
    recur(start_game_state)
    return memo

def minimax(
    game_state: GameState,
    #maximiser: Mark,
) -> tuple[int, tuple[GameState]]:
    res = minimaxFull(game_state)#, maximiser)
    return res[game_state]

def findBestMove(game_state: GameState) -> Move | None:
    #maximiser: Mark = game_state.current_mark
    best_score, best_score_moves = minimax(game_state)#, maximiser)
    #print(best_score, [move.after_state.grid.cells for move in best_score_moves])
    return random.choice(best_score_moves)

def findBestMovePrecomputed(game_state: GameState, filename: str=MinimaxSerialiser.DEFAULT_FILENAME) -> Move | None:
    optimal_moves = MinimaxSerialiser.loadGameStateOptimalMovesFromFile(game_state, filename=filename, create_if_not_exist=True)
    if optimal_moves is not None:
        best_score, best_score_moves = optimal_moves
        return random.choice(best_score_moves)
    optimal_moves = MinimaxSerialiser.loadGameStateOptimalMovesFromFile(
        game_state.inverted_game_state,
        filename=filename,
        create_if_not_exist=True
    )
    if optimal_moves is None:
        raise ValueError(
            "An unexpected error occurred where a game state was "
            "encountered for which neither itself or its inverted "
            "game state had been precomputed. This is most likely due "
            "to an error in the json file, so it is recommended to "
            "delete the file at filename and allow it to be "
            "reconstructed."
        )
    best_score, best_score_moves = optimal_moves
    #print("using inverse")
    #print(best_score, [move.after_state.grid.cells for move in best_score_moves])
    return random.choice(best_score_moves).inverted_move
    """
    print("Using findBestMovePrecomputed()")
    print("loading precomputed moves...")
    moves_dict = MinimaxSerialiser.load(filename=filename, create_if_not_exist=True)
    print("finished loading")
    optimal_moves = MinimaxSerialiser.loadGameStateOptimalMovesFromFile(game_state, filename=filename, create_if_not_exist=True)
    if game_state in moves_dict.keys():
        best_score, best_score_moves = moves_dict[game_state]
        #print(best_score, [move.after_state.grid.cells for move in best_score_moves])
        return random.choice(best_score_moves)
    game_state_inv = game_state.inverted_game_state
    best_score, best_score_moves = moves_dict[game_state_inv]
    print("using inverse")
    #print(best_score, [move.after_state.grid.cells for move in best_score_moves])
    return random.choice(best_score_moves).inverted_move
    """