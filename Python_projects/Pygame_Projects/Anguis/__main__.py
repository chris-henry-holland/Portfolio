#!/usr/bin/env python3
import os
import sys

#sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from game import Game


if __name__ == "__main__":
    game = Game(
        arena_shape=(15, 16),
        move_rate=15,
        n_fruit=1,
        head_init_direct=(0, 0)
    )
    game.run()
