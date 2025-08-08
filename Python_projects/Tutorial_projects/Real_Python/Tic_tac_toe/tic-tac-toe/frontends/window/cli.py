# frontends/window/cli.py

from __future__ import annotations

from queue import Queue
from threading import Thread

from tic_tac_toe.game.engine import TicTacToe

from tic_tac_toe.game.players import Player, RandomComputerPlayer, MinimaxComputerPlayer
from tic_tac_toe.logic.models import Mark

from .players import WindowPlayer
from .renderers import Window, WindowRenderer


def main() -> None:
    events: Queue = Queue()
    window = Window(events)
    Thread(
        target=gameLoop,
        args=(window,),
        kwargs={
            "events": events,
            "player1_type": WindowPlayer,
            "player2_type": MinimaxComputerPlayer,
        }, 
        daemon=True
    ).start()
    window.mainloop()

def gameLoop(
    window: Window,
    events: Queue,
    player1_type: type[Player]=WindowPlayer,
    player2_type: type[Player]=MinimaxComputerPlayer,
) -> None:
    player1 = player1_type(Mark("X"), events=events)
    player2 = player2_type(Mark("O"), events=events)
    starting_mark = Mark("X")
    TicTacToe(player1, player2, WindowRenderer(window)).play(starting_mark)
    return