# frontends/play.py

from __future__ import annotations

from window.cli import playTicTacToeWindow

from tic_tac_toe.game.players import PlayerType

if __name__ == "__main__":
    playTicTacToeWindow(
        player1_type=PlayerType.HUMAN,
        player2_type=PlayerType.MINIMAX,
    )

    #player1 = ConsolePlayer(Mark("X"))#RandomComputerPlayer(Mark("X"))
    #player2 = MinimaxComputerPlayer(Mark("O"))
    #
    #TicTacToe(player1, player2, WindowRenderer()).play()