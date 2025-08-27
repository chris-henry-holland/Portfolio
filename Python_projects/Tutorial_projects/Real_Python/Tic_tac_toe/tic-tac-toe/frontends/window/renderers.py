# frontends/window/renderers.py

from __future__ import annotations

import tkinter as tk

from tkinter import font, ttk
from queue import Queue

from tic_tac_toe.game.renderers import Renderer
from tic_tac_toe.logic.models import GameState

TFont_button = ("Arial", 36)
TFont_display = ("Arial", 32)

class TicTacToeWindowBoard(tk.Tk):

    def __init__(self, events: Queue, player_colors: dict[str, str]={"X": "blue", "O": "green"}):
        super().__init__()
        self.title("Tic-Tac-Tie Game")
        self.events = events
        self.player_colors = player_colors
        self._reset_requested = False
        #self._cells = {}
        self._buttons = []
        self._createMenu()
        self._createBoardDisplay()
        self._createBoardGrid()
        return

    def _createBoardDisplay(self) -> None:
        display_frame = tk.Frame(master=self)
        display_frame.pack(fill=tk.X)
        self.display = tk.Label(
            master=display_frame,
            text="Ready?",
            font=font.Font(font=TFont_display, weight="bold"),
        )
        self.display.pack()
        return
    
    def _createBoardGrid(self) -> None:
        grid_frame = tk.Frame(master=self)
        grid_frame.pack()
        for row in range(3):
            self.rowconfigure(row, weight=1, minsize=50)
            self.columnconfigure(row, weight=1, minsize=75)
            for col in range(3):
                button = tk.Button(
                    master=grid_frame,
                    text="",
                    font=font.Font(font=TFont_button, weight="bold"),
                    fg="black",
                    width=3,
                    height=2,
                    highlightbackground="lightblue",
                )
                self._buttons.append(button)
                #self._cells[button] = (row, col)
                button.bind("<ButtonRelease-1>", self.onButtonRelease)
                button.grid(
                    row=row,
                    column=col,
                    padx=5,
                    pady=5,
                    sticky="nsew",
                )
    
    def requestReset(self) -> None:
        #print("using requestReset()")
        self.events.put(-1)
        self._reset_requested = True
        return

    def resetBoard(self) -> None:
        self._updateDisplay(msg="Ready?")
        for button in self._buttons:
            button.config(highlightbackground="lightblue")
            button.config(text="")
            button.config(fg="black")
        self._reset_requested = False
        #print(self._reset_requested)
        return

    def onButtonRelease(self, event):
        clicked_button = event.widget
        self.events.put(self._buttons.index(clicked_button))
        #self._updateButton(self, clicked_button)
    
    def _configureButton(self, idx: int, label: str) -> None:
        button = self._buttons[idx]
        button.config(text=label)
        if label != " ":
            button.config(fg=self.player_colors[label])
            button.config(activeforeground=self.player_colors[label])
        return
    
    def _highlightCell(self, idx: int) -> None:
        self._buttons[idx].config(highlightbackground="red")
        return
    
    def _updateDisplay(self, msg: str, color: str="black") -> None:
        self.display["text"] = msg
        self.display["fg"] = color
        return
    
    def _createMenu(self) -> None:
        menu_bar = tk.Menu(master=self)
        self.config(menu=menu_bar)
        file_menu = tk.Menu(master=menu_bar)
        file_menu.add_command(
            label="Play Again",
            command=self.requestReset,
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=quit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        return
    
    #def isRequestingReset(self) -> bool:
    #    return self._reset_requested
    #
    #def resetCompleted(self) -> None:
    #    self._reset_requested = False

class TicTacToeWindowRenderer(Renderer):
    def __init__(self, window: TicTacToeWindowBoard) -> None:
        self.window = window
    
    def render(self, game_state: GameState) -> None:
        for idx, label, in enumerate(game_state.grid.cells):
            self.window._configureButton(idx, label)
        if game_state.tie:
            msg = "Tied game!"
            self.window._updateDisplay(msg=msg, color="red")
            return
        elif not game_state.winner:
            msg = f"{game_state.current_mark}'s turn"
            self.window._updateDisplay(msg)
            return
        msg = f"Player {game_state.winner} won!"
        self.window._updateDisplay(msg, color=self.window.player_colors[game_state.winner])
        #bold_style = ttk.Style()
        #bold_style.configure("Bold.TButton", font=("arial", 9, "bold"))
        for i in game_state.winning_cells:
            self.window._highlightCell(i)
        return
    
    def isResetRequested(self) -> bool:
        return self.window._reset_requested
    
    def reset(self) -> None:
        self.window.resetBoard()
        return


def main() -> None:
    """
    Create the tic-tac-toe game's board and run its main loop
    """
    board = TicTacToeWindowBoard()
    board.mainloop()

if __name__ == "__main__":
    main()