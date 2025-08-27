# frontends/console/cli.py

from tic_tac_toe.game.engine import TicTacToe



from .args import parseArgs
from .renderers import ConsoleRenderer

def main() -> None:
    player1, player2, starting_mark = parseArgs()
    while True:
        TicTacToe(player1, player2, ConsoleRenderer()).play(starting_mark)