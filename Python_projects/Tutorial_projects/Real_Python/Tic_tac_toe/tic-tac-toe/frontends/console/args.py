# frontends/console.args.py

import argparse

from typing import NamedTuple

from tic_tac_toe.game.players import (
    PlayerType,
    Player,
    RandomComputerPlayer,
    MinimaxComputerPlayer,
)
from tic_tac_toe.logic.models import Mark

from .players import ConsolePlayer

def loadPlayerType(player_type: PlayerType) -> type[Player]:
    if player_type == PlayerType.HUMAN:
        return ConsolePlayer
    elif player_type == PlayerType.RANDOM:
        return RandomComputerPlayer
    elif player_type == PlayerType.MINIMAX:
        return MinimaxComputerPlayer
"""
PLAYER_CLASSES = {
    PlayerType.HUMAN: ConsolePlayer,
    PlayerType.RANDOM: RandomComputerPlayer,
    PlayerType.MINIMAX: MinimaxComputerPlayer,
}
"""
class Args(NamedTuple):
    player1: Player
    player2: Player
    starting_mark: Mark

def parseArgs() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-X",
        dest="player_x",
        choices=PlayerType,
        type=PlayerType,
        default="human",
    )
    parser.add_argument(
        "-O",
        dest="player_o",
        choices=PlayerType,
        type=PlayerType,
        default="minimax",
    )
    parser.add_argument(
        "--starting",
        dest="starting_mark",
        choices=Mark,
        type=Mark,
        default="X",
    )
    args = parser.parse_args()

    player1 = loadPlayerType(args.player_x)(Mark("X"))#PLAYER_CLASSES[args.player_x](Mark("X"))
    player2 = loadPlayerType(args.player_o)(Mark("O"))#PLAYER_CLASSES[args.player_o](Mark("O"))

    if args.starting_mark == "O":
        player1, player2 = player2, player1
    
    return Args(player1, player2, args.starting_mark)