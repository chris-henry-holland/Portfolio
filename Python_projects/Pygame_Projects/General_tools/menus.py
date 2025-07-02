#!/usr/bin/env python

import functools
import os
import sys

from typing import Union, Tuple, List, Set, Dict, Optional, Callable, Any

import pygame as pg
import pygame.freetype

from pygame.locals import (
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

from .config import(
    enter_keys_def_glob,
    navkeys_def_glob,
    mouse_lclicks,
    named_colors_def,
    font_def_func,
)
from .utils import Real

from .user_input_processing import checkEvents, checkKeysPressed, getMouseStatus, createNavkeyDict, UserInputProcessor, UserInputProcessorMinimal
from .position_offset_calculators import topLeftFromAnchorPosition

from .buttons import ButtonGrid
from .text_manager import TextGroup

from .display_base_classes import InteractiveDisplayComponentBase

class MenuOverlay(InteractiveDisplayComponentBase):
    navkeys_def = navkeys_def_glob
    navkeys_dict_def = createNavkeyDict(navkeys_def)
    
    dynamic_displ_attrs = ["buttons"]
    static_bg_components = ["overlay_bg", "text"]

    def __init__(self, screen_shape: Tuple[Real]=(700, 700), framerate: Real=60,\
            overlay_color: Optional[Tuple[Union[Tuple[int], Real]]]=None,\
            mouse_enabled: bool=True, navkeys_enabled: bool=True,\
            navkeys: Optional[Tuple[Tuple[Set[int]]]]=None,\
            navkey_cycle_delay_s: Tuple[int]=(0.5, 0.2),\
            enter_keys: Optional[Set[int]]=None,\
            exit_press_keys: Optional[Set[int]]=None,\
            exit_release_keys: Optional[Set[int]]=None):
        
        super().__init__(screen_shape, (0, 0), anchor_type="topleft",\
            screen_topleft_offset=(0, 0),\
            mouse_enablement=(mouse_enabled, False, mouse_enabled),\
            navkeys_enablement=(navkeys_enabled, navkeys_enabled, False),\
            navkeys=navkeys,\
            enter_keys_enablement=(False, navkeys_enabled, False),\
            enter_keys=enter_keys)
        #print("Initializing menu overlay")
        
        # Resetting the user input processor
        self._user_input_processor = None
        
        self._screen_shape = screen_shape
        
        self._framerate = framerate
        
        self.overlay_color = overlay_color
        
        self.mouse_enabled = mouse_enabled
        self.navkeys_enabled = navkeys_enabled
        self.navkeys = navkeys_def_glob if navkeys is None else navkeys
        self._navkey_cycle_delay_s = navkey_cycle_delay_s
        self.setNavkeyCycleDelayFrame()
        
        #print("Setting exit_press_keys")
        self._exit_press_keys = set() if exit_press_keys is None else exit_press_keys
        #print(f"self._exit_press_keys = {self._exit_press_keys}")
        #print(f"self._user_input_processor = {self._user_input_processor}")
        #print("Finished setting exit_press_keys")
        self._exit_release_keys = set() if exit_release_keys is None else exit_release_keys
        self.buttons_uip_idx = None
        
        self.text_objects = []
        #self._text_img_constructors = []
        #self.text_groups = []
        #self.text_groups_args = []
        #self.text_groups_kwargs = []
        
        
    
    @property
    def exit_press_keys(self):
        return self._exit_press_keys
    
    @exit_press_keys.setter
    def exit_press_keys(self, exit_press_keys):
        #print("using setter")
        if exit_press_keys is None:
            exit_press_keys = set()
        #print(getattr(self, "_exit_press_keys", None))
        if exit_press_keys == getattr(self, "_exit_press_keys", None):
            return
        #print(f"setting exit_press_keys to {exit_press_keys}")
        self._exit_press_keys = exit_press_keys
        self._user_input_processor = None
    
    @property
    def exit_release_keys(self):
        return self._exit_release_keys
    
    @exit_release_keys.setter
    def exit_release_keys(self, exit_release_keys):
        #print("setter for exit_release_keys")
        if exit_release_keys is None:
            exit_release_keys = set()
        if exit_release_keys == getattr(self, "_exit_release_keys", None):
            return
        self._exit_release_keys = exit_release_keys
        self._user_input_processor = None
    
    
    @property
    def user_input_processor(self):
        res = getattr(self, "_user_input_processor", None)
        #print("getting user_input_processor")
        
        if res is None:
            keys_down_func=(lambda obj: set(self.navkeys_dict.keys()) if self.navkeys_enabled else set())
            key_press_event_filter = False if not self.exit_press_keys else\
                    (lambda obj, event: event.key in self.exit_press_keys)
            key_release_event_filter = False if not self.exit_release_keys else\
                    (lambda obj, event: event.key in self.exit_release_keys)
            #print("creating UserInputProcessor object")
            #print(f"exit press keys = {self.exit_press_keys}")
            res = UserInputProcessor(keys_down_func=keys_down_func,
                key_press_event_filter=key_press_event_filter,
                key_release_event_filter=key_release_event_filter,
                mouse_press_event_filter=False,
                mouse_release_event_filter=False,
                other_event_filter=False,
                get_mouse_status_func=False)
            self._user_input_processor = res
        return res
    
    
    @property
    def screen_shape(self):
        return self._screen_shape
    
    @screen_shape.setter
    def screen_shape(self, screen_shape):
        if screen_shape == getattr(self, "_screen_shape", None):
            return
        self._screen_shape = screen_shape
        self.screenShapeReset()
        return
    
    def screenShapeReset(self):
        self._resetButtonsSpatialProperties()
        self._resetTextGroupsSpatialProperties()
    
    @property
    def framerate(self):
        return self._framerate
    
    @framerate.setter
    def framerate(self, framerate):
        if framerate == self._framerate:
            return
        self._framerate = framerate
        self.setNavkeyCycleDelayFrame()
        return
    
    @property
    def navkeys(self):
        return self.navkeys_def if self._navkeys is None else self._navkeys
    
    @navkeys.setter
    def navkeys(self, navkeys):
        self._navkeys_dict = None
        self._navkeys = navkeys
        return
    
    @property
    def navkeys_dict(self):
        res = getattr(self, "_navkeys_dict", None)
        if res is None:
            navkeys = self.navkeys
            if navkeys is not None:
                res = self.getNavkeyDict(navkeys)
        return self.navkeys_dict_def if res is None else res
    
    @staticmethod
    def getNavkeyDict(navkeys: Tuple[Tuple[Set[int]]]):
        return createNavkeyDict(navkeys)
    
    @property
    def navkey_cycle_delay_s(self):
        return self._navkey_cycle_delay_s
    
    @navkey_cycle_delay_s.setter
    def navkey_cycle_delay_s(self, navkey_cycle_delay_s):
        if navkey_cycle_delay_s == self._navkey_cycle_delay_s:
            return
        self._navkey_cycle_delay_s = navkey_cycle_delay_s
        self.setNavkeyCycleDelayFrame()
        return
    
    @property
    def navkey_cycle_delay_frame(self):
        return self._navkey_cycle_delay_frame
    
    @staticmethod
    def timeTupleSecondToFrame(framerate: Real, time_tuple: Tuple[Real]) -> Tuple[int]:
        return tuple(round(x * framerate) for x in time_tuple)
    
    def setNavkeyCycleDelayFrame(self) -> None:
        res = self.timeTupleSecondToFrame(self._framerate, self._navkey_cycle_delay_s)
        self._navkey_cycle_delay_frame = res
        buttons = getattr(self, "buttons", None)
        if buttons is not None:
            buttons.navkey_cycle_delay_frame = res
        return
    
    def textPrinter(self, surf: "pg.Surface") -> None:
        surf_shape = (surf.get_width(), surf.get_height())
        for text_obj, max_shape_rel, anchor_pos_rel in self.text_objects:
            text_obj.max_shape = tuple(x * y for x, y in zip(surf_shape, max_shape_rel))
            text_obj.anchor_pos0 = tuple(x * y for x, y in zip(surf_shape, anchor_pos_rel))
        for text_obj, _, _ in self.text_objects:
            #anchor_pos = tuple(x * y for x, y in zip(surf_shape, anchor_pos_rel))
            text_obj.draw(surf)
        return
    
    def addText(self, text_obj: "Text", max_shape_rel: Tuple[Real],\
            anchor_pos_rel: Tuple[Real]) -> None:
        self.text_objects.append((text_obj, max_shape_rel, anchor_pos_rel))
        return
    """
    @property
    def text_surf(self):
        res = getattr(self, "_text_surf", None)
        if res is None:
            res = self.createTextSurface()
            self._text_surf = res
        return res
    
    def createTextSurface(self):
        surf = pg.Surface(self.screen_shape, pg.SRCALPHA)
        self.textPrinter(surf)
        return
    """
    #@property
    #def text_img_constructors(self):
    #    return self._text_img_constructors
    """
    def addTextGroup(self, text_list: List[Tuple[Union[str, Real]]],\
            max_height_rel: Real, font: Optional["pg.freetype"],\
            font_size: Optional[Real]=None) -> None:
        self.text_groups_args.append((text_list, max_height_rel, font))
        self.text_groups_kwargs.append({"font_size": font_size})
        max_height, text_list2 = self.findTextGroupsSpatialProperties(len(self.text_groups))
        
        self.text_groups.append(TextGroup(text_list2, max_height,\
                font, font_size=font_size, min_lowercase=True,\
                text_global_asc_desc_chars=None))
        return
    
    def findTextGroupsSpatialProperties(self, idx: int) -> Tuple[Union[Real, List[Tuple]]]:
        args = self.text_groups_args[idx]
        
        w, h = self.screen_shape
        max_height = args[1] * h
        #text_list = [("Hello", 200, (named_colors_def["red"], 1)), ...]
        text_list2 = [(x[0], x[3] * w, x[4]) for x in args[0]]
        return (max_height, text_list2)
    
    def _resetTextGroupsSpatialProperties(self) -> None:
        # TODO- implement more efficiently (so altering rather than
        # completely reset the Text objects
        for idx, text_group in enumerate(self.text_groups):
            max_height, text_list = self.findTextGroupsSpatialProperties(idx)
            text_group.max_height = max_height
            text_group.setupTextObjects(text_list)
        return
    """
    @property
    def buttons_spatial_props_rel(self):
        return getattr(self, "_buttons_spatial_props_rel", None)
    
    @property
    def button_spatial_props_rel(self, button_spatial_props_rel):
        prev = getattr(self, "_buttons_spatial_props_rel", None)
        if button_spatial_props_rel == prev:
            return
        self._resetButtonsSpatialProperties()
        return
    
    """
    @property
    def buttons_spatial_props(self):
        res = getattr(self, "_buttons_spatial_props", None)
        if res is None:
            res = self.findButtonsSpatialProperties()
            self._buttons_spatial_props = res
        return res
    """
    
    def findButtonsSpatialProperties(self) -> Tuple[Any]:
        bsp_rel = self.buttons_spatial_props_rel
        if bsp_rel is None:
            return ()
        (anchor_pos_rel, anchor_type, overall_shape_rel,\
                wh_ratio_range) = bsp_rel
        
        screen_shape = self.screen_shape
        anchor_pos = tuple(x * y for x, y in zip(screen_shape, anchor_pos_rel))
        
        overall_shape = [x * y for x, y in zip(screen_shape, overall_shape_rel)]
        #print(overall_shape)
        wh_ratio = overall_shape[0] / overall_shape[1]
        if wh_ratio < wh_ratio_range[0]:
            overall_shape[1] *= wh_ratio / wh_ratio_range[0]
        elif wh_ratio > wh_ratio_range[1]:
            overall_shape[0] *= wh_ratio_range[1] / wh_ratio
        
        overall_shape = tuple(overall_shape)
        topleft = topLeftFromAnchorPosition(overall_shape, anchor_type, anchor_pos)
        
        return (topleft, overall_shape)
    
    def _resetButtonsSpatialProperties(self) -> None:
        if not self.buttons: return
        bsp_rel = self.buttons_spatial_props_rel
        bsp = self.findButtonsSpatialProperties()
        if not bsp: return
        topleft, overall_shape = bsp
        self.buttons.dims = (*topleft, *overall_shape)
        return
    
    def setButtons(self, anchor_pos_rel: Tuple[Real], anchor_type: str,\
            overall_shape_rel: Tuple[Real], wh_ratio_range: Tuple[Real],\
            button_text_and_actions: List[List[Tuple[str, Any]]],\
            text_groups: Tuple[Union[Optional[Tuple["TextGroup"]], int]],\
            button_gap_rel_shape=(0.2, 0.2), fonts=None,\
            font_sizes=None, font_colors=None, text_borders_rel=None,\
            fill_colors=None, outline_widths=None,\
            outline_colors=None) -> None:
        
        self._buttons_spatial_props_rel = (anchor_pos_rel, anchor_type,\
                overall_shape_rel, wh_ratio_range)
        
        (topleft, overall_shape) = self.findButtonsSpatialProperties()
        
        button_text, button_actions = [], []
        for row in button_text_and_actions:
            button_text.append([])
            button_actions.append([])
            for text, action in row:
                button_text[-1].append(text)
                button_actions[-1].append(action)
        
        button_grid = ButtonGrid(overall_shape, button_text, text_groups,
            topleft, anchor_type="topleft",
            button_gap_rel_shape=button_gap_rel_shape,
            font_colors=font_colors, text_borders_rel=text_borders_rel,
            fill_colors=fill_colors, outline_widths=outline_widths,
            outline_colors=outline_colors,
            mouse_enabled=self.mouse_enabled,
            navkeys_enabled=self.navkeys_enabled,
            navkeys=self.navkeys,
            navkey_cycle_delay_frame=self.navkey_cycle_delay_frame)
        
        
        if self.buttons_uip_idx is not None:
            self.user_input_processor.removeSubUIP(self.buttons_uip_idx)
        self.buttons_uip_idx = self.user_input_processor.addSubUIP(button_grid.user_input_processor)
        """
        button_grid = ButtonGrid(topleft, (200, 50),
            button_text, button_gap_rel_shape=button_gap_rel_shape,
            fonts=fonts, font_sizes=font_sizes,
            font_colors=font_colors, text_borders_rel=text_borders_rel,
            fill_colors=fill_colors, outline_widths=outline_widths,
            outline_colors=outline_colors,
            mouse_enabled=self.mouse_enabled,
            navkeys_enabled=self.navkeys_enabled,
            navkeys=self.navkeys,
            navkey_cycle_delay_frame=self.navkey_cycle_delay_frame,
            text_global_asc_desc_chars=None)
        """
        
        #print(button_grid.mouse_enabled)
        #button_grid.grid_shape = overall_shape
        #button_grid.dims = (*topleft, *overall_shape)
        #print(button_grid.dims)
        self.buttons = button_grid
        self.button_actions = button_actions
        return
    
    @property
    def overlay_color(self):
        return self._overlay_color
    
    @overlay_color.setter
    def overlay_color(self, overlay_color):
        self._overlay_color = overlay_color
        self._overlay = None
        return
    
    @property
    def overlay_bg_surf(self):
        res = getattr(self, "_overlay_bg_surf", None)
        if res is None:
            res = self.createOverlayBackgroundSurface()
            self._overlay_bg_surf = res
        return res
    
    def createOverlayBackgroundSurface(self) -> "pg.Surface":
        if self.overlay_color is None: return ()
        overlay_bg_surf = pg.Surface(self.screen_shape, pg.SRCALPHA)
        color, alpha0 = self.overlay_color
        overlay_bg_surf.set_alpha(alpha0 * 255)
        overlay_bg_surf.fill(color)
        return overlay_bg_surf
    
    @property
    def overlay_bg_img_constructor(self):
        res = getattr(self, "_overlay_bg_img_constructor", None)
        if res is None:
            res = self.createOverlayBackgroundImageConstructor()
            self._overlay_bg_img_constructor = res
        return res
    
    def createOverlayBackgroundImageConstructor(self):
        overlay_bg_surf = self.overlay_bg_surf
        if overlay_bg_surf == ():
            return lambda surf: None
        return lambda surf: surf.blit(overlay_bg_surf, (0, 0))
    
    @property
    def text_surf(self):
        res = getattr(self, "_text_surf", None)
        if res is None:
            res = self.createTextSurface()
            self._text_surf = res
        return res
    
    def createTextSurface(self):
        if not self.text_objects:
            return ()
        surf = pg.Surface(self.screen_shape, pg.SRCALPHA)
        self.textPrinter(surf)
        return surf
    
    @property
    def text_img_constructor(self):
        res = getattr(self, "_text_img_constructor", None)
        if res is None:
            res = self.createTextImageConstructor()
            self._text_img_constructor = res
        return res
    
    def createTextImageConstructor(self):
        text_surf = self.text_surf
        if text_surf == ():
            return lambda surf: None
        return lambda surf: surf.blit(text_surf, (0, 0))
        
    @property
    def static_bg_surf(self):
        res = getattr(self, "_static_bg_surf", None)
        if res is None:
            res = self.createStaticBackgroundSurface()
            self._static_bg_surf = res
        return res
    
    def createStaticBackgroundSurface(self) -> None:
        #print("updating static background images")
        attrs = self.static_bg_components
        for attr in attrs:
            surf_attr = f"{attr}_surf"
            if getattr(self, surf_attr, None) != ():
                break
        else: return ()
        static_bg_surf = pg.Surface(self.screen_shape, pg.SRCALPHA)
        for attr in attrs:
            img_constr_attr = f"{attr}_img_constructor"
            getattr(self, img_constr_attr)(static_bg_surf)
        return static_bg_surf
    
    @property
    def static_bg_img_constructor(self):
        res = getattr(self, "_static_bg_img_constructor", None)
        if res is None:
            res = self.createStaticBackgroundImageConstructor()
            self._static_bg_img_constructor = res
        return res
    
    def createStaticBackgroundImageConstructor(self):
        #print("creating background constructor")
        static_bg_surf = self.static_bg_surf
        if static_bg_surf == ():
            return lambda surf: None
        return lambda surf: surf.blit(static_bg_surf, (0, 0))
    
    def draw(self, surf: "pg.Surface") -> None:
        shape = surf.get_width(), surf.get_height()
        self.screen_shape = shape
        
        self.static_bg_img_constructor(surf)
        """
        if self.overlay_bg_surf is not None:
            surf.blit(self.overlay_bg_surf, (0, 0))
        
        self.textPrinter(surf)
        """
        for attr in self.dynamic_displ_attrs:
            obj = getattr(self, attr, None)
            if obj is None: continue
            elif isinstance(obj, (list, tuple)):
                for obj2 in obj:
                    obj2.draw(surf)
            else:
                obj.draw(surf)
        return
    """
    @property
    def input_check_parameters(self):
        res = getattr(self, "_input_check_parameters", None)
        if res is None:
            res = self._findInputCheckParameters()
            self._input_check_parameters = res
        return res
    
    def _findInputCheckParameters(self):
        #print("hi")
        if hasattr(self, "buttons") and\
                hasattr(self.buttons, "input_check_parameters"):
            #print(self.buttons.input_check_parameters)
            return self.buttons.input_check_parameters
        return (None, None, None, False)
    """
    
    def getRequiredInputs(self):
        #print("Using MenuOverlay method getRequiredInputs()")
        quit, esc_pressed, events = self.user_input_processor.getEvents(self)
        return quit, esc_pressed, {"events": events,\
                "keys_down": self.user_input_processor.getKeysHeldDown(self),\
                "mouse_status": self.user_input_processor.getMouseStatus(self)}
    
    
        #quit, esc_pressed, events = self.user_input_processor.getEvents()
        #return quit, not esc_pressed, {"events": events,\
        #        "keys_down": self.user_input_processor.getKeysHeldDown(),\
        #        "mouse_status": self.user_input_processor.getMouseStatus()}
        """
        buttons = getattr(self, "buttons", None)
        if buttons is None:
            uip = UserInputProcessorMinimal()
            quit, esc_pressed, events = uip.getEvents()
            return quit, esc_pressed, {"events": events}
        return buttons.getRequiredInputs()
        """
        #extra_events, new_keys_to_check, keys_to_check, check_mouse = self.input_check_parameters
        #quit, running, events = checkEvents(extra_events, new_keys_to_check)
        #if quit or not running:
        #    return quit, running, events, None, None, None
        #keys_pressed = checkKeysPressed(self.navkeys_dict.keys())
        #mouse_pos, mouse_pressed = getMouseStatus() if check_mouse else (None, None)
        #return quit, running, events, keys_pressed, mouse_pos, mouse_pressed
    
    #def eventLoop(self, mouse_pos: Optional[Tuple[bool]]=None,\
    #        events: Optional[List[int]]=None,\
    #        keys_pressed: Optional[List[int]]=None,\
    #        mouse_pressed: Optional[Tuple[int]]=None,\
    #        check_axes: Tuple[int]=(0, 1))\
    #        -> Tuple[Union[bool, Tuple[int]]]:
    
    def eventLoop(self, events: Optional[List[int]]=None,\
            keys_down: Optional[Set[int]]=None,\
            mouse_status: Optional[Tuple[int]]=None,\
            check_axes: Tuple[int]=(0, 1))\
            -> Tuple[Union[bool, Tuple[Union[bool, Any]]]]:
        #print("using menu eventLoop() method")
        running, quit = True, False
        buttons = getattr(self, "buttons", None)
        screen_changed = False
        
        #print(mouse_status)
        #print(self.mouse_enablement, self.mouse_enabled)
        
        if events is None:
            #print(events)
            quit, esc_pressed, events = self.user_input_processor.getEvents()
            if esc_pressed:
                running = False
        #if events:
        #    print(events)
        #    print(self.exit_press_keys)
        #    print(self.exit_release_keys)
        events2 = []
        for event_tup in events:
            #print((event_tup[1] == 0 and event_tup[0].key in\
            #        self.exit_press_keys))
            if not (event_tup[1] == 0 and event_tup[0].key in\
                    self.exit_press_keys) and not (event_tup[1] == 1\
                    and event_tup[0].key in self.exit_release_keys):
                 events2.append(event_tup)
                 continue
            running = False
            break
        actions = []
        if buttons is not None:
            quit2, running2, (change, selected_b_inds) = buttons.eventLoop(\
                    events=events, keys_down=keys_down,\
                    mouse_status=mouse_status, check_axes=check_axes)
            if change: screen_changed = True
            if quit2: quit = True
            if not running2: running = False
            actions = [self.button_actions[b_inds[0]][b_inds[1]] for b_inds in selected_b_inds]
        #action = None if selected is None else\
        #        self.button_actions[selected[0]][selected[1]]
        return quit, running, (screen_changed, actions)
