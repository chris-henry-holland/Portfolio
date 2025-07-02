#!/usr/bin/env python
from collections import deque
import random
import sys
import os
import copy

from sortedcontainers import SortedSet, SortedDict

from typing import Union, Tuple, List, Set, Dict, Optional, Callable, Any

import pygame as pg
import pygame.freetype

from pygame.locals import (
    RLEACCEL,
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
    K_RETURN,
    K_KP_ENTER,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from General_tools import (
    checkEvents,
    checkKeysPressed,
    createNavkeyDict,
    getMouseStatus,
    Real,
    enter_keys_def_glob,
    navkeys_def_glob,
    mouse_lclicks,
    named_colors_def,
    font_def_func,
    MenuOverlay,
    TextGroup,
)

from gameplay import GamePlay

class Game:
    def __init__(self, head_size=25, arena_shape=(16, 15),
            head_init_pos=None, head_init_direct=(0, 1),
            move_rate=15, n_frame_per_move=2,
            n_fruit=1, border=((1, 1), (4, 1)),
            font=None, auto=False, auto_startpos=(1, 1),
            auto_fruitpos=(((-2, 1), (1, 0)), ((-2, -2), (0, 1)),
            ((1, -2), (-1, 0)), ((1, 1), (0, -1))),
            navkeys=None, menu_framerate: int=60):
        pg.init()
        self.head_size = head_size
        self.arena_shape = arena_shape
        self.head_init_pos = head_init_pos
        self.head_init_direct = head_init_direct
        self.move_rate = move_rate
        self.n_frame_per_move = n_frame_per_move
        self.n_fruit = n_fruit
        self.border = border
        self.menu_framerate = menu_framerate
        self.font = font_def_func() if font is None else font
        self.auto_startpos = auto_startpos
        
        # Note that index 1 indicates the direction the snake should
        # turn when encounters this fruit
        self.auto_fruitpos_prov = auto_fruitpos
        
        
        self.playing = False
        self.quit = False
        #self.first_screen = SnakeGame
        
        
        self.pause_fr = 30
        self.death_screen_fr = 30
        self.font_sizes = (3, 1.8, 1.2) # relative to head_size
        self.pause_overlay_color = (named_colors_def["gray"], 0.5)
        self.death_screen_overlay_color = (named_colors_def["red"], 0.5)
        self.menu_overlay_color = (named_colors_def["green"], 1)
        self.button_border = 0.4 # relative to head_size
        self.menu_arrow_cycle_delay_s = (0.5, 0.1)
        
        self.enter_keys = enter_keys_def_glob
        self.menu_nav_keys = ({K_DOWN}, {K_UP})
    
    @property
    def gameplay(self):
        res = getattr(self, "_gameplay", None)
        if res is None:
            res = GamePlay(screen=self.screen, head_size=self.head_size, arena_shape=self.arena_shape,
                    head_init_pos=self.head_init_pos, head_init_direct=self.head_init_direct,
                    move_rate=self.move_rate, n_frame_per_move=self.n_frame_per_move,
                    n_fruit=self.n_fruit, border=self.border,
                    font=self.font, auto=False, navkeys=None)
            self._gameplay = res
        return res
        
    def getMenuOverlay(self, mouse_enabled: bool=True, navkeys_enabled: bool=True) -> "MenuOverlay":
    
        menu_overlay = MenuOverlay(self.screen_shape, framerate=self.menu_framerate,\
            overlay_color=self.menu_overlay_color,\
            mouse_enabled=navkeys_enabled, navkeys_enabled=navkeys_enabled)
        
        max_height_rel = 0.2
        max_width_rel = 0.8
        anchor_type = "midbottom"
        anchor_pos_rel = (0.5, 0.3)
        font_color = (named_colors_def["white"], 1)
        
        text_group = TextGroup([], max_height0=None, font=None, font_size=None, min_lowercase=True, text_global_asc_desc_chars=None)
        text_list = [
            ({"text": "Main menu", "font_color": font_color, "anchor_type0": anchor_type}, ((max_width_rel, max_height_rel), anchor_pos_rel)),
        ]
        
        add_text_list = [x[0] for x in text_list]
        text_objs = text_group.addTextObjects(add_text_list)
        #print(text_objs)
        for text_obj, (_, pos_tup) in zip(text_objs, text_list):
            max_shape_rel, anchor_pos_rel = pos_tup
            menu_overlay.addText(text_obj, max_shape_rel,\
                    anchor_pos_rel)
        
        button_text_groups = tuple((TextGroup([], max_height0=None, font=None, font_size=None, min_lowercase=True, text_global_asc_desc_chars=None),) for _ in range(4))
        #button_text_and_actions =\
        #        [[(("Play game", "center"), (lambda: (0, True, False, True, False))),\
        #        (("Options", "center"), (lambda: (-1, True, False, True, False))),\
        #        (("Exit", "center"), (lambda: (-1, True, False, False, True)))]]
        button_anchor_pos_tup = ("center", 0, 0, 0)
        button_text_and_actions =\
                [[(("Play game", button_anchor_pos_tup), 1),\
                (("Watch bot", button_anchor_pos_tup), 2),\
                (("Options", button_anchor_pos_tup), 3),\
                (("Exit", button_anchor_pos_tup), 0)]]
        
        menu_overlay.setButtons((0.5, 0.35), "midtop",\
            (0.5, 0.55), wh_ratio_range=(0.1, 10),\
            button_text_and_actions=button_text_and_actions,\
            text_groups=button_text_groups,\
            button_gap_rel_shape=(0.1, 0.1),\
            font_colors=((named_colors_def["white"], 0.5), (named_colors_def["yellow"], 1), (named_colors_def["blue"], 1), (named_colors_def["green"], 1)),
            text_borders_rel=((0.2, 0.2), (0.1, 0.1), 1, 0),\
            fill_colors=(None, (named_colors_def["red"], 0.2), (named_colors_def["red"], 0.5), 2),\
            outline_widths=((1,), (2,), (3,), 1),\
            outline_colors=((named_colors_def["black"], 1), (named_colors_def["blue"], 1), 1, 1))
        return menu_overlay
    
    @property
    def menu_overlay(self):
        res = getattr(self, "_menu_overlay", None)
        if res is None:
            res = self.getMenuOverlay()
            self._menu_overlay = res
        return res
    
    @property
    def border(self):
        return self._border
    
    @border.setter
    def border(self, border):
        self._arena_ul = None
        self._border = border
    
    @property
    def head_size(self):
        return self._head_size
    
    @head_size.setter
    def head_size(self, head_size):
        self._arena_dims = None
        self._screen_shape = None
        self._arena_ul = None
        self._head_size = head_size
    
    @property
    def arena_shape(self):
        return self._arena_shape
    
    @arena_shape.setter
    def arena_shape(self, arena_shape):
        self._arena_dims = None
        self._screen_shape = None
        self._auto_fruitpos = None
        self._arena_ul = None
        self._arena_shape = arena_shape
    
    @property
    def arena_dims(self):
        arena_dims = getattr(self, "_arena_dims", None)
        if arena_dims is not None:
            return arena_dims
        self._arena_dims = tuple(self._head_size * x for x in\
                                self._arena_shape)
        return self._arena_dims
    
    @property
    def arena_ul(self):
        # Position of the upper left corner of the arena
        arena_ul = getattr(self, "_arena_ul", None)
        if arena_ul is not None:
            return arena_ul
        self._arena_ul = ActualPosition([x[0] for x, y in\
                        zip(self.border, self.arena_shape)],\
                        scale=self.head_size)
        return self._arena_ul
    
    @property
    def screen_shape(self):
        screen_shape = getattr(self, "_screen_shape", None)
        if screen_shape is not None:
            return screen_shape
        self._screen_shape = tuple(self.head_size * (x + sum(y))\
                for x, y in zip(self.arena_shape, self.border))
        
        return self._screen_shape
    
    @property
    def screen(self):
        screen = getattr(self, "_screen", None)
        if screen is None:
            self._screen = pg.display.set_mode(self.screen_shape)
            pg.display.set_caption("Anguis")
        return self._screen
    
    @property
    def arena(self):
        arena = getattr(self, "_arena", None)
        if arena is not None:
            return arena
        self._arena = pg.draw.rect(self.screen, self.white,\
                        [*self.arena_ul, *self.arena_dims])
        return self._arena
    
    
    @property
    def auto_fruitpos(self):
        auto_fruitpos = getattr(self, "_auto_fruitpos", None)
        if auto_fruitpos is not None:
            return auto_fruitpos
        auto_fruitpos = []
        for fruitpos_prov in self.auto_fruitpos_prov:
            auto_fruitpos.append([[]])
            for x, y in zip(fruitpos_prov[0], self.arena_shape):
                if x >= 0:
                    auto_fruitpos[-1][0].append(x)
                    continue
                #print(auto_fruitpos[-1])
                #print(auto_fruitpos[-1][0])
                auto_fruitpos[-1][0].append(x + y)
            auto_fruitpos[-1][0] = tuple(auto_fruitpos[-1][0])
            auto_fruitpos[-1].append(fruitpos_prov[1])
        self._auto_fruitpos = tuple(auto_fruitpos)
        return self._auto_fruitpos
    """
    def prep_menu(self, menu_name):
        font_sizes = (2.5, 1.2)
        text_dict_list = []
        text_colors = (self.black, self.white)
        if menu_name.strip().lower() == "main_menu":
            text_dict_list = [{"text": "Main menu", "font": self.font,\
                "font_size": font_sizes[0],\
                "text_offset": (0, -4),\
                "offset_anchors": ("center", "center"),\
                "ref_obj": self.screen.get_rect(),\
                "text_color": text_colors[0]}]
            text_dict_list.append({"text": "Play game", "font": self.font,\
                "font_size": font_sizes[1],\
                "text_offset": (0, -1),\
                "offset_anchors": ("center", "center"),\
                "ref_obj": self.screen.get_rect(),\
                "text_color": text_colors[1], "button_group": 0,\
                "action": lambda: self.main_menu.run_gameplay()})
            text_dict_list.append({"text": "Options", "font": self.font,\
                "font_size": font_sizes[1],\
                "text_offset": (0, 1),\
                "offset_anchors": ("center", "center"),\
                "ref_obj": self.screen.get_rect(),\
                "text_color": text_colors[1], "button_group": 0})
            text_dict_list.append({"text": "High scores", "font": self.font,\
                "font_size": font_sizes[1],\
                "text_offset": (0, 3),\
                "offset_anchors": ("center", "center"),\
                "ref_obj": self.screen.get_rect(),\
                "text_color": text_colors[1], "button_group": 0})
            text_dict_list.append({"text": "Exit", "font": self.font,\
                "font_size": font_sizes[1],\
                "text_offset": (0, 5),\
                "offset_anchors": ("center", "center"),\
                "ref_obj": self.screen.get_rect(),\
                "text_color": text_colors[1], "button_group": 0,\
                "action": lambda: self.main_menu.quit()})
        
        menu = Menus(self, text_dict_list,\
            frm_rate=self.menu_frm_rate,\
            overlay_color=self.menu_overlay_color,\
            overlay_opacity=self.menu_overlay_opacity,\
            button_border=self.button_border,\
            nav_keys=self.menu_nav_keys)
        setattr(self, menu_name, menu)
        return menu
    """
    def menuActionResolver(self, action: int) -> Tuple[bool]:
        if not action:
            return (True, True, False, True)
        elif action > 2:
            return (False, False, True, False)
        
        score, retry, quit = self.gameplay.run(auto=(action == 2))
        if not retry:
            return (True, True, not quit, quit)
        return self.menuActionResolver(action)
    
    def actionResolver(self, overlay_attr: str, action: int) -> Tuple[bool]:
        if overlay_attr == "menu_overlay":
            return self.menuActionResolver(action)
        return (False, False, True, False)
        """
        if overlay_attr != "menu_overlay" or action_output != 0:
            return (-1, False, False, True, False)
        gameplay = GamePlay(screen=self.screen, head_size=self.head_size, arena_shape=self.arena_shape,
                head_init_pos=self.head_init_pos, head_init_direct=self.head_init_direct,
                move_rate=self.move_rate, n_frame_per_move=self.n_frame_per_move,
                n_fruit=self.n_fruit, border=self.border,
                font=self.font, auto=False, navkeys=None)
        score, retry, quit = gameplay.run()
        if quit:
            return (-1, True, True, False, True)
        if retry:
            return (0, True, True, True, False)
        return (-1, True, True, True, False)
        """
    
    def menuOverlay(self, overlay_attr: str) -> bool:
        
        screen = self.screen
        screen_cp = pg.Surface.copy(screen)
        overlay = getattr(self, overlay_attr)
        framerate = overlay.framerate
        restart = False
        quit = False
        screen_changed = True
        
        clock = pg.time.Clock()
        while True:
            quit, esc_pressed, event_loop_kwargs = overlay.getRequiredInputs()
            running = not esc_pressed
            if quit or not running:
                break
            change = False
            quit2, running2, (chng, actions) = overlay.eventLoop(check_axes=(0, 1), **event_loop_kwargs)
            if not running2: running = False
            if quit2: quit = True
            for action in actions:
                #print(action)
                ignore_subsequent, chng2, running2, quit2 = self.actionResolver(overlay_attr, action)
                
                #action_output, ignore_subsequent, chng2, running2, quit2 = action()
                #if chng2: chng = True
                #while action_output != -1 and running2 and not quit2:
                #    print(action_output)
                #    action_output, ignore_subsequent2, chng2, running2, quit2 = self.actionOutputResolver(overlay_attr, action_output)
                #    if chng2: chng = True
                #    if ignore_subsequent2: ignore_subsequent = True
                #print(f"actions: restart = {restart2}, running = {running2}, quit = {quit2}")
                #if restart2: restart = True
                if chng2: chng = True
                if not running2: running = False
                if quit2: quit = True
                if ignore_subsequent: break
            if quit or not running:
                break
            if chng: change = True
            if change: screen_changed = True
            if screen_changed:
                screen.blit(screen_cp, (0, 0))
                overlay.draw(screen)
                pg.display.flip()
            clock.tick(framerate)
            screen_changed = False
        return restart, quit
    
    def menu(self) -> bool:
        return self.menuOverlay(overlay_attr="menu_overlay")
    
    def run(self):
        return self.menu()

if __name__ == "__main__":
    game = Game(move_rate=15, n_fruit=1, head_init_direct=(0, 0))
    game.run()
