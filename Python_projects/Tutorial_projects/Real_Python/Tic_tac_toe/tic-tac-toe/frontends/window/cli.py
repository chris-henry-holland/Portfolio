# frontends/window/cli.py

from __future__ import annotations

from queue import Queue
from threading import Thread

from tic_tac_toe.game.engine import TicTacToe

from tic_tac_toe.game.players import Player, RandomComputerPlayer, MinimaxComputerPlayer, PlayerType
from tic_tac_toe.logic.models import Mark

from .players import WindowPlayer
from .renderers import TicTacToeWindowBoard, TicTacToeWindowRenderer

def loadPlayerType(player_type: PlayerType) -> type[Player]:
    if player_type == PlayerType.HUMAN:
        return WindowPlayer
    elif player_type == PlayerType.RANDOM:
        return RandomComputerPlayer
    elif player_type == PlayerType.MINIMAX:
        return MinimaxComputerPlayer

def playTicTacToeWindow(
    player1_type: PlayerType=PlayerType.HUMAN,
    player2_type: PlayerType=PlayerType.MINIMAX,
) -> None:
    events: Queue = Queue()
    window = TicTacToeWindowBoard(events)
    player1_cls = loadPlayerType(player1_type)
    player2_cls = loadPlayerType(player2_type)
    Thread(
        target=gameLoop,
        args=(window,),
        kwargs={
            "events": events,
            "player1_cls": player1_cls,
            "player2_cls": player2_cls,
        }, 
        daemon=True
    ).start()
    window.mainloop()
    return

def gameLoop(
    window: TicTacToeWindowBoard,
    events: Queue,
    player1_cls: type[Player]=WindowPlayer,
    player2_cls: type[Player]=MinimaxComputerPlayer,
) -> None:
    player1 = player1_cls(Mark("X"), events=events)
    player2 = player2_cls(Mark("O"), events=events)
    starting_mark = Mark("X")
    while True:
        TicTacToe(player1, player2, TicTacToeWindowRenderer(window)).play(starting_mark)
    return

def main() -> None:
    playTicTacToeWindow(
        player1_type=PlayerType.HUMAN,
        player2_type=PlayerType.MINIMAX,
    )
    return