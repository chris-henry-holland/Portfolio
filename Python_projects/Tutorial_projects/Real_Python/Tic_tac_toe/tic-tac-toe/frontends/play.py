# frontends/play.py

from __future__ import annotations

from tic_tac_toe.game.engine import TicTacToe
from tic_tac_toe.game.players import RandomComputerPlayer, MinimaxComputerPlayer
from tic_tac_toe.logic.models import Mark

from console.players import ConsolePlayer
from console.renderers import ConsoleRenderer
from window.renderers import WindowRenderer

if __name__ == "__main__":
    player1 = ConsolePlayer(Mark("X"))#RandomComputerPlayer(Mark("X"))
    player2 = MinimaxComputerPlayer(Mark("O"))

    TicTacToe(player1, player2, WindowRenderer()).play()