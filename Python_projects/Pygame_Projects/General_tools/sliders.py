#!/usr/bin/env python
import math

from typing import Any, Tuple, Union, List, Optional, Callable

import pygame as pg

from .config import (
    named_colors_def,
    font_def_func,
)
from .utils import Real, ColorOpacity

from .display_base_classes import (
    InteractiveDisplayComponentBase,
    ComponentGroupBaseClass,
    ComponentGroupElementBaseClass,
    checkHiddenKwargs,
)
from .text_manager import Text, TextGroup
from .position_offset_calculators import topLeftAnchorOffset

track_color_def = (named_colors_def["silver"], 1)
thumb_color_def = (named_colors_def["white"], 1)
text_color_def = (named_colors_def["black"], 1)

def sliderValuesInteger(obj: "Slider") -> bool:
    return isinstance(obj.increment_start, int) and bool(obj.increment) and isinstance(obj.increment, int)

def sliderPrimaryDemarcationsInteger(obj: "Slider") -> bool:
    res = sliderValuesInteger(obj) and isinstance(obj.demarc_start_val, int) and (obj.__dict__.get("_demarc_intervals", None) is None or isinstance(obj.demarc_intervals[0], int))
    #print(f"demarcations integer = {res}")
    return res

def sliderDemarcationsIntervalDefault(obj: "Slider") -> Real:
    pow10 = math.floor(math.log(obj.val_range[1] - obj.val_range[0], 10) - 0.5)
    return (10 ** pow10,) if sliderPrimaryDemarcationsInteger(obj) or pow10 > 1 else (1,)

def sliderDemarcationsDPDefault(obj: "Slider") -> int:
    if sliderPrimaryDemarcationsInteger(obj):
        return 0
    res = max(0, -math.floor(math.log(obj.demarc_intervals[0], 10) - 1.5))
    return res

def sliderPlusDemarcationsDPDefault(obj: "SliderPlus") -> int:
    return sliderDemarcationsDPDefault(obj.slider)

class Slider(InteractiveDisplayComponentBase):
    slider_names = set()
    unnamed_count = 0
    
    reset_graph_edges = {
        "topleft_screen": {"thumb_x_screen": True, "slider_ranges_screen": True},
        #"mouse_enabled": {"mouse_enablement": True},
        "track_color": {"track_surf": True},
        "track_topleft": {"x_range": True, "static_bg_surf": True, "demarc_surf": True, "slider_ranges_surf": True},
        "track_shape": {"track_surf": True, "x_range": True, "demarc_surf": True, "slider_ranges_surf": True},
        "demarc_line_colors": {"demarc_surf": True},
        
        "demarc_numbers_color": {"demarc_surf": True},
        
        "increment_start": {"val_range_actual": True},
        "increment": {"val_range_actual": True},
        "val_range": {"val_range_actual": True},
        "val_range_actual": {"x_range": True},
        "x_range": {"thumb_x": True},
        
        "thumb_radius": {"thumb_surf": True, "slider_ranges_surf": True},
        "thumb_color": {"thumb_surf": True},
        
        "val_raw": {"val": True},
        "val": {"thumb_x": True},
        "thumb_x_screen_raw": {"val": True},
        "thumb_x_screen": {"thumb_x": True},
        
        "slider_ranges_surf": {"slider_ranges_screen": True},
        
        "track_surf": {"static_bg_surf": True},
        "demarc_surf": {"static_bg_surf": True},
        "static_bg_surf": {"display_surf": True},
        "thumb_x": {"display_surf": True, "thumb_x_screen": True},
        "thumb_surf": {"display_surf": True},
    }
    
    component_dim_determiners = ["shape", "demarc_numbers_max_height_rel", "demarc_numbers_dp", "val_range", "demarc_intervals", "demarc_start_val", "thumb_radius_rel", "demarc_line_lens_rel", "demarc_numbers_text_group", "demarc_numbers_text_objects"]
    dim_dependent = ["track_shape", "track_topleft", "demarc_numbers_height", "thumb_radius"]
    
    for attr1 in component_dim_determiners:
        reset_graph_edges.setdefault(attr1, {})
        for attr2 in dim_dependent:
            reset_graph_edges[attr1][attr2] = True
    
    custom_reset_methods = {
        "val_raw": "setValueRaw",
    }
    
    attribute_calculation_methods = {
        "mouse_enablement": "calculateMouseEnablement",
        "track_topleft": "calculateAndSetComponentDimensions",
        "track_shape": "calculateAndSetComponentDimensions",
        "thumb_radius": "calculateAndSetComponentDimensions",
        "demarc_numbers_height": "calculateAndSetComponentDimensions",
        "val_range_actual": "calculateValueRangeActual",
        "x_range": "calculateXRange",
        "val": "calculateValue",
        "thumb_x": "calculateThumbX",
        "thumb_x_screen": "calculateThumbXScreen",
        "slider_ranges_surf": "calculateSliderRangesSurface",
        "slider_ranges_screen": "calculateSliderRangesScreen",
        
        "static_bg_surf": "createStaticBackgroundSurface",
        "track_surf": "createTrackSurface",
        "demarc_numbers_text_objects": "createDemarcationNumbersTextObjects",
        "demarc_surf": "createDemarcationSurface",
        "thumb_surf": "createThumbSurface",
        "display_surf": "createDisplaySurface",
        
        "track_img_constructor": "createTrackImageConstructor",
        "demarc_img_constructor": "createDemarcationsImageConstructor",
        "static_bg_img_constructor": "createStaticBackgroundImageConstructor",
        "thumb_img_constructor": "createThumbImageConstructor",
    }
    
    @staticmethod
    def demarcNumsTextGroup():
        #print("\ncreating TextGroup for the final Slider object")
        res = Slider.createDemarcationNumbersTextGroup(max_height=None)#self.demarc_numbers_height)
        #print("finished creating TextGroup for final Slider object")
        return res
    
    attribute_default_functions = {
        "increment": ((lambda obj: 0),),
        "val_raw": ((lambda obj: -float("inf")),),
        "demarc_numbers_text_group": ((lambda obj: Slider.demarcNumsTextGroup()),),#createDemarcationNumbersTextGroup()),),
        "demarc_numbers_dp": (sliderDemarcationsDPDefault, ("demarc_intervals", "demarc_start_val", "increment", "increment_start")),
        "thumb_radius_rel": ((lambda obj: 1),),
        "demarc_line_lens_rel": ((lambda obj: tuple(0.5 ** i for i in range(10))),),
        "demarc_intervals": (sliderDemarcationsIntervalDefault, ("val_range", "increment", "increment_start")),
        "demarc_start_val": ((lambda obj: obj.val_range[0]), ("val_range",)),
        "demarc_numbers_max_height_rel": ((lambda obj: 2),),
        
        "demarc_numbers_color": ((lambda obj: text_color_def),),
        "track_color": ((lambda obj: track_color_def),),
        "demarc_line_colors": ((lambda obj: (obj.track_color,)), ("track_color",)),
        "thumb_color": ((lambda obj: thumb_color_def),),
        "thumb_outline_color": ((lambda obj: ()),),
        
        "mouse_enabled": ((lambda obj: True),),
    }
    
    fixed_attributes = set()
    static_bg_components = ["track", "demarc"]
    displ_component_attrs = ["static_bg", "thumb"]
    
    def __init__(self,
        shape: Tuple[Real],
        anchor_pos: Tuple[Real],
        val_range: Tuple[Real],
        increment_start: Real,
        increment: Optional[Real]=None,
        anchor_type: Optional[str]=None,
        screen_topleft_offset: Optional[Tuple[Real]]=None,
        init_val: Optional[Real]=None,
        demarc_numbers_text_group: Optional["TextGroup"]=None,
        demarc_numbers_dp: Optional[int]=None,
        thumb_radius_rel: Optional[Real]=None,
        demarc_line_lens_rel: Optional[Tuple[Real]]=None,
        demarc_intervals: Optional[Tuple[Real]]=None,
        demarc_start_val: Optional[Real]=None,
        demarc_numbers_max_height_rel: Optional[Real]=None,
        track_color: Optional[ColorOpacity]=None,
        thumb_color: Optional[ColorOpacity]=None,
        demarc_numbers_color: Optional[ColorOpacity]=None,
        demarc_line_colors: Optional[ColorOpacity]=None,
        thumb_outline_color: Optional[ColorOpacity]=None,
        mouse_enabled: Optional[bool]=None,
        name: Optional[str]=None,
        **kwargs,
    ) -> None:
        checkHiddenKwargs(type(self), kwargs)
        if name is None:
            Slider.unnamed_count += 1
            name = f"slider {self.unnamed_count}"
        #self.name = name
        Slider.slider_names.add(name)
        super().__init__(**self.initArgsManagement(locals(), kwargs=kwargs, rm_args=["init_val"]), slider_held=False, val_raw=init_val)
    
    #def setMouseEnabled(self, prev_val: bool) -> None:
    #    #print("using method setMouseEnablement()")
    #    mouse_enabled = self.mouse_enabled
    #    self.mouse_enablement = (mouse_enabled, mouse_enabled, mouse_enabled)
    #    #print(f"self.mouse_enablement = {self.mouse_enablement}")
    #    return
    
    def calculateMouseEnablement(self) -> None:
        #print("calculating mouse enablement")
        mouse_enabled = self.mouse_enabled
        return (mouse_enabled, mouse_enabled, mouse_enabled)
    
    def setValueRaw(self, prev_val: Real) -> None:
        self.__dict__["_thumb_x_screen_raw"] = None
        return
    
    def calculateValueRangeActual(self) -> Tuple[Real]:
        increment = self.increment
        increment_start = self.increment_start
        val_range = self.val_range
        if not increment:
            return val_range
        vra = [val_range[0] if increment_start is None else\
                increment_start + math.ceil((val_range[0] - increment_start) / increment) * increment]
        vra.append(vra[0] + math.floor((val_range[1] - vra[0]) / increment) * increment)
        return tuple(vra)
    
    def calculateXRange(self) -> Tuple[int]:
        return tuple(map(self.val2X, self.val_range_actual))
    
    def calculateValue(self) -> Real:
        #print("calculating value")
        thumb_x_screen_raw = self.__dict__.get("_thumb_x_screen_raw", None)
        if thumb_x_screen_raw is not None:
            return self.x2Val(thumb_x_screen_raw - self.topleft_screen[0])
        return self.findNearestValue(self.val_raw)
    
    def calculateThumbX(self) -> int:
        return self.val2X(self.val)
    
    def calculateThumbXScreen(self) -> int:
        return () if self.thumb_x == () else self.thumb_x + self.topleft_screen[0]
    
    #def createDemarcationNumbersTextGroup(self, max_height: Optional[Real]=None) -> "TextGroup":
    #    text_group = self.__dict__.get("_demarc_numbers_text_group", None)
    #    font = None if text_group is None else text_group.font
    #    return self._createDemarcationNumbersTextGroupGivenFontAndMaxHeight(font=font, max_height=max_height)
    
    @staticmethod
    def createDemarcationNumbersTextGroup(
        font: Optional["pg.freetype"]=None,
        max_height: Optional[int]=None,
    ) -> "TextGroup":
        #print("hello")
        dig_txt_lst = [str(d) for d in range(10)]
        dig_txt_lst.append(".")
        text_list = [{"text": "".join(dig_txt_lst)}]
        if font is None:
            font = font_def_func()
        #text_list = [{"text": str(d)} for d in range(10)]
        return TextGroup(text_list,\
                max_height0=max_height, font=font,\
                font_size=None, min_lowercase=True,\
                text_global_asc_desc_chars=None)
    
    def createDemarcationNumbersTextGroupCurrentFont(
        self,
        max_height: Optional[int]=None,
    ) -> "TextGroup":
        text_group = self.__dict__.get("_demarc_numbers_text_group", None)
        font = None if text_group is None else text_group.font
        return self.createDemarcationNumbersTextGroup(font=font, max_height=max_height)
    
    def findNearestValue(self, val: Real):
        vra = self.val_range_actual
        if val <= vra[0]:
            return vra[0]
        elif val >= vra[1]:
            return vra[1]
        elif not self.increment:
            return val
        return vra[0] + round((val - vra[0]) / self.increment) * self.increment
    
    def _calculateComponentDimensions(self, text_obj_lists: List[List[Tuple["TextGroupElement", Real]]]) -> Tuple[Union[Tuple[Real], Real]]:
        #print("using _calculateComponentDimensions()")
        y0 = 0
        h0 = self.shape[1]
        h_ratio = self.demarc_numbers_max_height_rel + (3 * self.demarc_line_lens_rel[0] / 2) + 1 + max(self.thumb_radius_rel - 0.5, 0)
        h = math.floor(h0 / h_ratio)
        y = y0 + math.ceil(h * max(self.thumb_radius_rel - 0.5, 0))
        thumb_radius = math.ceil(h * self.thumb_radius_rel)
        
        min_end_gaps = (thumb_radius, thumb_radius)
        min_demarc_numbers_gap = 2
        
        #def maxXTrackDimensions(text_obj_lists: List[List[Tuple["TextGroupElement", Real]]], text_h: int) -> int:
        #    return self._maxXTrackDimensionsGivenTextObjects(text_obj_lists, min_gaps=min_gaps)
        
        text_h_max = math.floor(self.demarc_numbers_max_height_rel * h)
        #print(f"text_h_max = {text_h_max}")
        def textHeightAllowed(text_h: int) -> bool:
            for text_objs in text_obj_lists:
                track_width = self.maxXTrackDimensionsGivenTextHeight(text_h, min_gaps=min_end_gaps)[1]
                if self.numbersOverlapGivenTrackWidthAndTextHeight(text_objs, track_width, text_h, min_gap=min_demarc_numbers_gap):
                    return False
            return True
        lft, rgt = 0, text_h_max
        while lft < rgt:
            mid = lft - ((lft - rgt) >> 1)
            if textHeightAllowed(mid):
                lft = mid
            else: rgt = mid - 1
        gaps, w = self.maxXTrackDimensionsGivenTextHeight(lft, min_gaps=min_end_gaps)
        x = gaps[0]
        return ((w, h), (x, y), thumb_radius, lft)
    
    #def calculateTrackDimensions(self) -> Tuple[Real]:
    def calculateComponentDimensions(self) -> Tuple[Union[Tuple[Real], Real]]:
        #print("using calculateComponentDimensions()")
        #print("creating demarcation numbers text group")
        #print("\ncreating TextGroup to calculate Slider component dimensions")
        text_group = self.createDemarcationNumbersTextGroupCurrentFont()
        #print("finished creating TextGroup to calculate Slider component dimensions")
        #print("creating demarcation numbers text objects")
        #print("creating TextGroup elements to calculate Slider component dimensions")
        text_objs = self._createDemarcationNumbersTextObjectsGivenTextGroupAndMaxHeight(demarc_numbers_text_group=text_group, max_height=None)
        #print("finished creating TextGroup elements to calculate Slider component dimensions")
        #print("finished creating demarcation numbers text objects")
        res = self._calculateComponentDimensions([text_objs])
        #print(f"component dimensions = {res}")
        return res
        
    
    def setTrackDimensions(self, shape: Tuple[int], topleft: Tuple[int]) -> None:
        self._track_shape = shape
        self._track_topleft = topleft
        return
    
    def setThumbRadius(self, thumb_radius: int) -> None:
        self._thumb_radius = thumb_radius
        return
    
    def setDemarcationNumbersHeight(self, height: int) -> None:
        self._demarc_numbers_height = height
        text_objs = getattr(self, "_demarc_numbers_text_objects", None)
        if text_objs is None:
            return
        for text_obj_tup in text_objs:
            text_obj_tup[0].max_height = height
        return
    
    def setComponentDimensions(self, track_shape: Tuple[int], track_topleft: Tuple[int], thumb_radius: int, text_height: int) -> None:
        self.setTrackDimensions(track_shape, track_topleft)
        self.setThumbRadius(thumb_radius)
        self.setDemarcationNumbersHeight(text_height)
        return
    
    #def calculateAndSetTrackDimensions(self) -> None:
    def calculateAndSetComponentDimensions(self) -> Optional[Tuple[Union[Tuple[Real], Real]]]:
        res = self.calculateComponentDimensions()
        self.setComponentDimensions(*res)
        return res
    
    def _maxXTrackDimensionsGivenTextObjects(self, text_obj_lists: List[List[Tuple["Text", Real]]], min_gaps: Tuple[int]=(0, 0)) -> Tuple[Union[Tuple[int], int]]:
        mx_w = self.shape[0]
        
        text_end_pairs = []
        for text_objs in text_obj_lists:
            if not text_objs: continue
            text_obj1, val1 = text_objs[0]
            text_obj2, val2 = text_objs[-1]
            text_end_pairs.append(((text_obj1, val1, -text_obj1.calculateTopleftEffective((0, 0), anchor_type="midtop")[0]),\
                    (text_obj2, val2, text_obj2.calculateTopleftEffective((0, 0), anchor_type="midtop")[0] + text_obj2.shape_eff[0])))
        if not text_end_pairs:
            gaps = tuple(math.ceil(gap) for gap in min_gaps)
            return (gaps, mx_w - sum(gaps))
        
        
        def trackEndGaps(track_width: int) -> Tuple[int]:
            gaps = list(min_gaps)
            for (text_obj1, val1, gap1), (text_obj2, val2, gap2) in text_end_pairs:
                gaps[0] = max(gaps[0], gap1 - self.val2TrackLeftDistGivenTrackWidth(val1, track_width))
                gaps[1] = max(gaps[1], gap2 - track_width + self.val2TrackLeftDistGivenTrackWidth(val2, track_width))
            return tuple(math.ceil(gap) for gap in gaps)
        
        lft, rgt = 0, mx_w - sum(min_gaps)
        while lft < rgt:
            mid = lft - ((lft - rgt) >> 1)
            gaps = trackEndGaps(mid)
            #print(mid, gaps)
            if mid + sum(gaps) <= mx_w:
                lft = mid
            else: rgt = mid - 1
        gaps = trackEndGaps(lft)
        
        return (gaps, lft)
    
    def maxXTrackDimensionsGivenTextHeight(self, text_height: int, min_gaps: Tuple[int]=(0, 0)) -> Tuple[Union[Tuple[int], int]]:
        #print("\nUsing maxXTrackDimensionsGivenTextHeight()")
        #print(f"text_height = {text_height}")
        #print("\ncreating TextGroup to calculate Slider x-dimension for a given text height")
        text_group = self.createDemarcationNumbersTextGroupCurrentFont(max_height=text_height)
        #print("finished creating TextGroup to calculate Slider x-dimension for a given text height")
        #print("creating TextGroup elements to calculate Slider x-dimension for a given text height")
        text_obj_lists = [self._createDemarcationNumbersTextObjectsGivenTextGroupAndMaxHeight(demarc_numbers_text_group=text_group, max_height=None)]
        #print("finished creating TextGroup elements to calculate Slider x-dimension for a given text height")
        res = self._maxXTrackDimensionsGivenTextObjects(text_obj_lists, min_gaps=min_gaps)
        #print("finished calculating Slider x-dimension")
        return res
    
    def numbersOverlapGivenTrackWidthAndTextHeight(self, text_objs: List[Tuple["Text", Real]], track_width: int, text_height: int, min_gap: int=2) -> bool:
        if not text_objs:
            return False
        text_group = text_objs[0][0].text_group
        text_group.max_height0 = text_height
        curr_right = -float("inf")
        for text_obj_tup in text_objs:
            text_obj, val = text_obj_tup
            curr_left = text_obj.calculateTopleftEffective((self.val2TrackLeftDistGivenTrackWidth(val, track_width), 0), anchor_type="midtop")[0]
            if curr_left <= curr_right + min_gap:
                return True
            curr_right = curr_left + text_obj.shape_eff[0]
        return False
    """
    def maxTextHeightGivenTrackWidth(self, track_width: int, text_height_max: Optional[int]=None, min_gap: int=2) -> Union[int, bool]:
        #if not self.demarc_numbers_text_objects:
        #    return 0
        if text_height_max is None:
            text_height_max = self.shape[1]
        
        print("\ncreating TextGroup to calculate max Slider text height for a given track width")
        text_group = self.createDemarcationNumbersTextGroupCurrentFont()
        print("finished creating TextGroup to calculate max Slider text height for a given track width")
        print("creating TextGroup elements to calculate max Slider text height for a given track width")
        text_objs = self.createDemarcationNumbersTextObjects(demarc_numbers_text_group=text_group)
        print("finished creating TextGroup elements to calculate max Slider text height for a given track width")
        if not text_objs: return 0
        
        def numbersOverlap(text_height: int) -> bool:
            return self.numbersOverlapGivenTrackWidthAndTextHeight(text_objs, track_width, text_height, min_gap=min_gap)
        
        if not numbersOverlap(text_height_max):
            return text_height_max
        
        lft, rgt = 0, text_height_max
        while lft < rgt:
            #print()
            mid = lft - ((lft - rgt) >> 1)
            #print(lft, rgt, mid)
            if numbersOverlap(mid):
                rgt = mid - 1
            else: lft = mid
        #print(f"text_height = {lft}")
        return lft
    """
    def createTrackSurface(self) -> "pg.Surface":
        #print("creating track surface")
        if not self.track_color: return ()
        shape = [x + 1 - i for i, x in enumerate(self.track_shape)]
        track_surf = pg.Surface(shape, pg.SRCALPHA)
        track_surf.set_alpha(self.track_color[1] * 255)
        track_surf.fill(self.track_color[0])
        return track_surf
    
    def createTrackImageConstructor(self) -> Callable[["pg.Surface"], None]:
        return lambda obj, surf: (None if obj.track_surf == () else surf.blit(obj.track_surf, obj.track_topleft))
    
    
    def resetDemarcationNumbersTextObjects(self) -> None:
        text_objs = getattr(self, "_demarc_numbers_text_objects", None)
        if not text_objs: return
        self.demarc_numbers_text_group.removeTextObjects([x[0] for x in text_objs])
        self._demarc_numbers_text_objects = None
        return
    
    def createDemarcationNumbersTextObjects(self)\
            -> List[Tuple["Text", Real]]:
        #print("using createDemarcationNumbersTextObjects()")
        demarc_numbers_text_group = self.demarc_numbers_text_group
        max_height = self.demarc_numbers_height
        #print(f"max_height = {max_height}")
        res = self._createDemarcationNumbersTextObjectsGivenTextGroupAndMaxHeight(demarc_numbers_text_group, max_height=max_height, displ_text=True)
        #if res:
        #    print(f"text object reset functions = {getattr(res[0][0], '_attr_reset_funcs', None)}")
        return res
    
    def _createDemarcationNumbersTextObjectsGivenTextGroupAndMaxHeight(
        self,
        demarc_numbers_text_group: "TextGroup",
        max_height: Optional[Real]=None,
        displ_text: bool=False,
    ) -> List[Tuple["Text", Real]]:
        #print("creating demarcation numbers text objects")
        if not self.demarc_intervals:
            return []
        #if demarc_numbers_text_group is None:
        #    demarc_numbers_text_group = self.demarc_numbers_text_group
        intvl = self.demarc_intervals[0]
        val_start = self.demarc_start_val + math.ceil((self.val_range[0] - self.demarc_start_val) / intvl) * intvl
        #print(f"val_start = {val_start}")
        val = val_start
        add_text_dicts = []
        dp = self.demarc_numbers_dp
        color = self.demarc_numbers_color
        vals = []
        def updateFunction(obj: "Slider", prev_val: Optional["Text"]) -> None:
            #print("updated text")
            setattr(obj, "demarc_surf", None)
            return
        
        while val <= self.val_range[1]:
            val_txt = f"{val:.{dp}f}"
            add_text_dict = {"text": val_txt, "font_color": color}
            if max_height is not None:
                add_text_dict["max_shape"] = (None, max_height)
            if displ_text:
                #print("hi")
                add_text_dict["_attr_reset_funcs"] = {"updated": [updateFunction]}
            add_text_dicts.append(add_text_dict)
            vals.append(val)
            val += self.demarc_intervals[0]
        #print(add_text_dicts)
        #print(demarc_numbers_text_group.max_height)
        res = list(zip(demarc_numbers_text_group.addTextObjects(add_text_dicts), vals))
        #print("added text objects")
        #print(demarc_numbers_text_group.max_height)
        return res
    
    def createDemarcationSurface(self) -> Union["pg.Surface", tuple]:
        #print("creating demarcation surface")
        demarc_intvls = self.demarc_intervals
        if not demarc_intvls: return ()
        line_lens_rel = self.demarc_line_lens_rel
        #if not line_lens_rel: return ()
        surf = pg.Surface(self.shape, pg.SRCALPHA)
        
        line_upper_y = self.track_topleft[1] + self.track_shape[1]
        line_colors = self.demarc_line_colors
        line_lens = tuple(round(self.track_shape[1] * x) for x in line_lens_rel)
        
        numbers_color = self.demarc_numbers_color
        numbers_upper_y = int(line_upper_y + line_lens[0] * 1.5)
        numbers_height = self.demarc_numbers_height #self.demarc_number_size_rel * self.track_shape[1]
        
        seen_x = set()
        
        def lineRenderer(line_idx: int) -> List[int]:
            if not line_lens_rel: return []
            line_surf = pg.Surface(self.shape, pg.SRCALPHA)
            line_x_vals = []
            line_color = line_colors[min(line_idx, len(line_colors) - 1)] if line_colors else ()
            line_len = line_lens[min(line_idx, len(line_lens) - 1)] if line_lens else 0
            intvl = demarc_intvls[line_idx]
            val = self.demarc_start_val + math.ceil((self.val_range[0] - self.demarc_start_val) / intvl) * intvl
            while val <= self.val_range[1]:
                line_x = round(self.val2X(val))
                line_x_vals.append(line_x)
                if line_len and line_color and line_x not in seen_x:
                    seen_x.add(line_x)
                    line_y_rng = [round(line_upper_y + dy) for dy in (0, line_len)]
                    #print(f"drawing line {[(line_x, y) for y in line_y_rng]} with color {line_color[0]}")
                    pg.draw.line(line_surf, line_color[0], *[(line_x, y) for y in line_y_rng])
                val += intvl
            line_surf.set_alpha(255 * line_color[1])
            surf.blit(line_surf, (0, 0))
            return line_x_vals
            
        x_lst = lineRenderer(0)
        for x, text_obj_tup in zip(x_lst, self.demarc_numbers_text_objects):
            text_obj = text_obj_tup[0]
            anchor_pos = (x, numbers_upper_y)
            #print(f"numbers_height = {numbers_height}")
            text_obj.max_shape = (None, numbers_height)
            #print(f"text_obj shape = {text_obj.shape}")
            text_obj.draw(surf, anchor_pos=anchor_pos, anchor_type="midtop")
        
        for line_idx in range(1, len(demarc_intvls)):
            lineRenderer(line_idx)
    
        return surf
    
    def createDemarcationsImageConstructor(self)\
            -> Callable[["pg.Surface"], None]:
        return lambda obj, surf: (None if obj.demarc_surf == () else surf.blit(obj.demarc_surf, (0, 0)))
    
    def createStaticBackgroundSurface(self)\
            -> Union["pg.Surface", tuple]:
        #print("creating static background surface")
        surf = pg.Surface(self.shape, pg.SRCALPHA)
        surf.set_alpha(255)
        #surf.fill((255, 0, 0))
        
        constructor_attrs = [f"{attr}_img_constructor"\
                for attr in self.static_bg_components]
        #print(constructor_attrs)
        for constructor_attr in constructor_attrs:
            
            constructor_func =\
                    getattr(self, constructor_attr, (lambda obj, surf: None))
            #print(constructor_attr)
            constructor_func(self, surf)
        return surf
    
    def createStaticBackgroundImageConstructor(self)\
            -> Callable[["pg.Surface"], None]:
        return lambda obj, surf: (None if obj.static_bg_surf == () else surf.blit(obj.static_bg_surf, (0, 0)))
        
    def createThumbSurface(self) -> Union["pg.Surface", tuple]:
        thumb_color = self.thumb_color
        if not thumb_color:
            return ()
        rad = self.thumb_radius
        surf = pg.Surface(self.shape, pg.SRCALPHA)
        pg.draw.circle(surf, thumb_color[0], (rad, rad), rad)
        surf.set_alpha(thumb_color[1] * 255)
        return surf
    
    def createThumbImageConstructor(self)\
            -> Callable[["pg.Surface"], None]:
        def thumbRenderer(obj: "Slider", surf: "pg.Surface") -> None:
            if obj.thumb_surf == () or obj.thumb_x == ():
                return
            topleft = (obj.thumb_x - obj.thumb_radius, obj.track_topleft[1] + (obj.track_shape[1] / 2) - obj.thumb_radius)
            #print(f"thumb topleft = {topleft}")
            surf.blit(obj.thumb_surf, topleft)
            return
        return thumbRenderer
    
    def createDisplaySurface(self) -> Optional["pg.Surface"]:
        #print("\ncreating display surface")
        surf = pg.Surface(self.shape, pg.SRCALPHA)
        for attr in self.displ_component_attrs:
            #print(f"{attr}_img_constructor")
            constructor_func = getattr(self, f"{attr}_img_constructor", (lambda obj, surf: None))
            #print(f"  {attr}, {surf}")
            constructor_func(self, surf)
        #print(f"surf = {surf}")
        return surf
    
    def draw(self, surf: "pg.Surface") -> None:
        #print("\ndrawing slider")
        #print("display_surf" in self.__dict__.keys())
        #print(self.__dict__.get("display_surf", None))
        #print("_display_surf" in self.__dict__.keys())
        #print(self.__dict__.get("_display_surf", None))
        #print(f"self.display_surf = {self.display_surf}")
        #print("display_surf" in self.__dict__.keys())
        #print(self.__dict__.get("display_surf", None))
        #print("_display_surf" in self.__dict__.keys())
        #print(self.__dict__.get("_display_surf", None))
        surf.blit(self.display_surf, self.topleft)
        return
    
    def val2TrackLeftDistGivenTrackWidth(self, val: Real, track_width: int) -> int:
        return round(track_width * (val - self.val_range[0]) / (self.val_range[1] - self.val_range[0]))
    
    def val2X(self, val: Union[Real, tuple]) -> Union[int, tuple]:
        if isinstance(val, tuple): return ()
        return self.track_topleft[0] +\
                self.val2TrackLeftDistGivenTrackWidth(val, self.track_shape[0])
    
    def x2Val(self, x):
        x_range = self.x_range
        vra = self.val_range_actual
        if x < x_range[0]:
            return vra[0]
        elif x > x_range[1]:
            return vra[1]
        elif not self.increment:
            return vra[0] + ((vra[1] - vra[0]) * ((x - x_range[0]) / float(x_range[1] - x_range[0])))
        return vra[0] + round((vra[1] - vra[0]) * ((x - x_range[0]) / (self.increment * (x_range[1] - x_range[0])))) * self.increment
    
    def calculateSliderRangesSurface(self) -> Tuple[Tuple[Real]]:
        #print("finding slider ranges surface")
        y_extend = max(1, self.thumb_radius_rel - 1) * self.track_shape[1]
        x_extend = self.thumb_radius_rel * self.track_shape[1]
        return ((self.track_topleft[0] - x_extend, self.track_topleft[0] + self.track_shape[0] + x_extend), (self.track_topleft[1] - y_extend, self.track_topleft[1] + self.track_shape[1] + y_extend))
    
    def calculateSliderRangesScreen(self) -> Tuple[Tuple[Real]]:
        return self.rangesSurface2RangesScreen(self.slider_ranges_surf)
    
    def mouseOverSlider(self, mouse_pos: Tuple[int], check_axes: Tuple[int]=(0, 1)):
        #print("checking mouseOverSlider()")
        rngs = self.slider_ranges_screen
        #print(rngs, mouse_pos)
        #print(mouse_pos, rngs, self.slider_ranges_surf, self.screen_topleft_offset)
        return all(rngs[i][0] <= mouse_pos[i] <= rngs[i][1] for i in check_axes)
    
    def processEvents(self, events: List[Tuple[int]]) -> List[Tuple[int]]:
        res = []
        for event_tup in events:
            if 2 <= event_tup[1] <= 3 and event_tup[0].button == 1:
                res.append((event_tup[0].pos, event_tup[1]))
        return res
    
    def eventLoop(self, events: Optional[List[int]]=None,\
            keys_down: Optional[List[int]]=None,\
            mouse_status: Optional[Tuple[int]]=None,\
            check_axes: Tuple[int]=(0, 1))\
            -> Tuple[bool, bool, bool, Any]:
        #print("calling Slider eventLoop()")
        #print(events)
        ((quit, esc_pressed), (events, keys_down, mouse_status, check_axes)) = self.getEventLoopArguments(events=events, keys_down=keys_down, mouse_status=mouse_status, check_axes=check_axes)
        running = not quit and not esc_pressed
        screen_changed = False
        
        """
        quit = False
        running = True
        screen_changed = False
        
        if events is None:
            quit, esc_pressed, events = user_input_processor.getEvents()
            if esc_pressed or quit:
                running = False
        
        if mouse_status is None:
            mouse_status = user_input_processor.getMouseStatus() if self.mouse_enablement[0] else ()
        """
        slider_held = self.slider_held
        thumb_x_screen_raw_changed = False
        for event_tup in self.processEvents(events):
            if event_tup[1] == 2:
                slider_held = self.mouseOverSlider(event_tup[0], check_axes=check_axes)
                #print(f"slider_held = {slider_held}")
            elif event_tup[1] == 3 and slider_held:
                thumb_x_screen_raw_changed = True
                thumb_x_screen_raw = event_tup[0][0]
                slider_held = False
        #print(mouse_status)
        slider_held = slider_held and mouse_status and mouse_status[1][0]
        #print(f"slider_held 2 = {slider_held}")
        if slider_held:
            thumb_x_screen_raw_changed = True
            thumb_x_screen_raw = mouse_status[0][0]
            
        self.slider_held = slider_held
        if thumb_x_screen_raw_changed:
            thumb_x0 = self.thumb_x
            self.thumb_x_screen_raw = thumb_x_screen_raw
            screen_changed = (self.thumb_x != thumb_x0)
            
        else: screen_changed = False
        return quit, running, screen_changed, self.val

class SliderGroupElement(ComponentGroupElementBaseClass, Slider):
    
    group_cls_func = lambda: SliderGroup
    group_obj_attr = "slider_group"
    #fixed_attributes = {group_obj_attr}
    
    def __init__(
        self,
        slider_group: "SliderGroup",
        anchor_pos: Tuple[Real],
        val_range: Tuple[Real],
        increment_start: Real,
        increment: Optional[Real]=None,
        anchor_type: Optional[str]=None,
        screen_topleft_offset: Optional[Tuple[Real]]=None,
        init_val: Optional[Real]=None,
        demarc_numbers_dp: Optional[int]=None,
        demarc_intervals: Optional[Tuple[Real]]=None,
        demarc_start_val: Optional[Real]=None,
        name: Optional[str]=None,
        **kwargs,
    ) -> None:
        
        checkHiddenKwargs(type(self), kwargs)
        
        #self.__dict__[f"_{self.group_obj_attr}"] = slider_group
        super().__init__(
            shape=slider_group.slider_shape,
            anchor_pos=anchor_pos,
            val_range=val_range,
            increment_start=increment_start,
            increment=increment,
            anchor_type=anchor_type,
            screen_topleft_offset=screen_topleft_offset,
            init_val=init_val,
            demarc_numbers_text_group=slider_group.demarc_numbers_text_group,
            demarc_numbers_dp=demarc_numbers_dp,
            thumb_radius_rel=slider_group.thumb_radius_rel,
            demarc_line_lens_rel=slider_group.demarc_line_lens_rel,
            demarc_intervals=demarc_intervals,
            demarc_start_val=demarc_start_val,
            demarc_numbers_max_height_rel=slider_group.demarc_numbers_max_height_rel,
            track_color=slider_group.track_color,
            thumb_color=slider_group.thumb_color,
            demarc_numbers_color=slider_group.demarc_numbers_color,
            demarc_line_colors=slider_group.demarc_line_colors,
            thumb_outline_color=slider_group.thumb_outline_color,
            mouse_enabled=slider_group.mouse_enabled,
            name=name,
            _group=slider_group,
            **kwargs,
        )
    
    
    def calculateComponentDimensions(self) -> Tuple[Union[Tuple[Real], Real]]:
        #print("\ncreating TextGroup to calculate SliderGroup component dimensions")
        text_group = self.createDemarcationNumbersTextGroupCurrentFont()
        #print("finished creating TextGroup to calculate SliderGroup component dimensions")
        #print("creating TextGroup elements to calculate SliderGroup component dimensions")
        text_obj_lists = [slider_weakref()._createDemarcationNumbersTextObjectsGivenTextGroupAndMaxHeight(demarc_numbers_text_group=text_group, max_height=None) for slider_weakref in self.slider_group._elements_weakref]
        #print("finished creating TextGroup elements to calculate SliderGroup component dimensions")
        return self._calculateComponentDimensions(text_obj_lists)
    
    def setComponentDimensions(self, track_shape: Tuple[int], track_topleft: Tuple[int], thumb_radius: int, text_height: int) -> None:
        for cls2 in type(self).mro()[1:]:
            if "setComponentDimensions" in cls2.__dict__.keys():
                ancestor_method = cls2.setComponentDimensions
                break
        else: return
        if ancestor_method is None:
            return
        for slider_weakref in self.slider_group._elements_weakref:
            ancestor_method(slider_weakref(), track_shape, track_topleft, thumb_radius, text_height)
        return
    
    def maxXTrackDimensionsGivenTextHeight(self, text_height: int, min_gaps: Tuple[int]=(0, 0)) -> Tuple[Union[Tuple[int], int]]:
        #print("\ncreating TextGroup to calculate SliderGroup x-dimension for a given text height")
        text_group = self.createDemarcationNumbersTextGroupCurrentFont(max_height=text_height)
        #print("finished creating TextGroup to calculate SliderGroup x-dimension for a given text height")
        #print("creating TextGroup elements to calculate SliderGroup x-dimension for a given text height")
        text_obj_lists = []
        for slider_weakref in self.slider_group._elements_weakref:
            text_obj_lists.append(slider_weakref()._createDemarcationNumbersTextObjectsGivenTextGroupAndMaxHeight(demarc_numbers_text_group=text_group, max_height=None))
        #print("finished creating TextGroup elements to calculate SliderGroup x-dimension for a given text height")
        return self._maxXTrackDimensionsGivenTextObjects(text_obj_lists, min_gaps=min_gaps)

class SliderGroup(ComponentGroupBaseClass):
    group_element_cls_func = lambda: SliderGroupElement
    
    reset_graph_edges = {}
    
    """
    custom_reset_methods = {
        "slider_shape": "setSliderShape",
        "demarc_numbers_text_group": "setDemarcationNumbersTextGroup",
        "thumb_radius_rel": "setThumbRadiusRelative",
        "demarc_line_lens_rel": "setDemarcationLineLengthsRelative",
        "demarc_numbers_max_height_rel": "setDemarcationNumbersMaxHeightRelative",
        "track_color": "setTrackColor",
        "thumb_color": "setThumbColor",
        "demarc_numbers_color": "setDemarcationNumbersColor",
        "demarc_line_colors": "setDemarcationLineColors",
        "thumb_outline_color": "setThumbOutlineColor",
        "mouse_enabled": "setMouseEnabled",
    }
    """
    attribute_calculation_methods = {}
    
    # Review- account for using element_inherited_attributes in ComponentGroupBaseClass
    attribute_default_functions = {
        attr: Slider.attribute_default_functions.get(attr) for attr in
        [
            "demarc_numbers_text_group",
            "thumb_radius_rel",
            "demarc_line_lens_rel",
            "demarc_numbers_max_height_rel",
            "track_color",
            "thumb_color",
            "demarc_numbers_color",
            "demarc_line_colors",
            "thumb_outline_color",
            "mouse_enabled",
        ]
    }
    
    #fixed_attributes = {"sliders"}
    
    element_inherited_attributes = {
        "slider_shape": "shape",
        "demarc_numbers_text_group": "demarc_numbers_text_group",
        "thumb_radius_rel": "thumb_radius_rel",
        "demarc_line_lens_rel": "demarc_line_lens_rel",
        "demarc_numbers_max_height_rel": "demarc_numbers_max_height_rel",
        "track_color": "track_color",
        "demarc_numbers_color": "demarc_numbers_color",
        "demarc_line_colors": "demarc_line_colors",
        "thumb_outline_color": "thumb_outline_color",
        "mouse_enabled": "mouse_enabled",
    }
    
    def __init__(self, 
        slider_shape: Tuple[Real],
        demarc_numbers_text_group: Optional["TextGroup"]=None,
        thumb_radius_rel: Optional[Real]=None,
        demarc_line_lens_rel: Optional[Tuple[Real]]=None,
        demarc_numbers_max_height_rel: Optional[Real]=None,
        track_color: Optional[ColorOpacity]=None,
        thumb_color: Optional[ColorOpacity]=None,
        demarc_numbers_color: Optional[ColorOpacity]=None,
        demarc_line_colors: Optional[ColorOpacity]=None,
        thumb_outline_color: Optional[ColorOpacity]=None,
        mouse_enabled: Optional[bool]=None,
        **kwargs,
    ) -> None:
        checkHiddenKwargs(type(self), kwargs)
        super().__init__(**self.initArgsManagement(locals(), kwargs=kwargs))
        
    
    def addSlider(
        self,
        anchor_pos: Tuple[Real],
        val_range: Tuple[Real],
        increment_start: Real,
        increment: Optional[Real]=None,
        anchor_type: Optional[str]=None,
        screen_topleft_offset: Optional[Tuple[Real]]=None,
        init_val: Optional[Real]=None,
        demarc_numbers_dp: Optional[int]=None,
        demarc_intervals: Optional[Tuple[Real]]=None,
        demarc_start_val: Optional[Real]=None,
        name: Optional[str]=None,
        **kwargs,
    ) -> "SliderGroupElement":
        
        res = self._addElement(
            anchor_pos=anchor_pos,
            val_range=val_range,
            increment_start=increment_start,
            increment=increment,
            anchor_type=anchor_type,
            screen_topleft_offset=screen_topleft_offset,
            init_val=init_val,
            demarc_numbers_dp=demarc_numbers_dp,
            demarc_intervals=demarc_intervals,
            demarc_start_val=demarc_start_val,
            name=name,
        )
        return res

class SliderPlus(InteractiveDisplayComponentBase):
    sliderplus_names = set()
    unnamed_count = 0
    
    reset_graph_edges = {
        "shape": {"slider_shape": True, "slider_bottomleft": True, "title_shape": True, "val_text_shape": True},
        "slider_shape_rel": {"slider_shape": True},
        "slider_borders_rel": {"slider_borders": True},
        "slider_shape": {"title_shape": True},
        "slider_borders": {"title_shape": True},
        "title_shape": {"title_anchor_pos": (lambda obj: obj.title_anchor_type != "topleft"), "title_surf": True},
        "title_anchor_type": {"title_anchor_pos": True, "title_surf": True},

        "val": {"val_str": True},
        "val_str": {"val_text_surf": True},

        "val_text_shape": {"val_text_anchor_pos": (lambda obj: obj.val_text_anchor_type != "topleft"), "val_text_surf": True},
        "val_text_anchor_type": {"val_text_anchor_pos": True, "val_text_surf": True},
        
        "title_surf": {"static_bg_surf": True},
        "val_text_surf": {"display_surf": True},
        "static_bg_surf": {"display_surf": True},
    }
    
    custom_reset_methods = {
        "title_shape": "setTitleShape",
    }
    
    attribute_calculation_methods = {
        #"slider": "createSlider",
        "slider_shape": "calculateSliderShape",
        "slider_bottomleft": "calculateSliderBottomLeft",
        "slider_borders": "calculateSliderBorders",

        "val": "calculateValue",
        "val_str": "calculateValueString",

        "title_text_group": "createTitleTextGroup",
        "title_shape": "calculateTitleShape",
        "title_anchor_pos": "calculateTitleAnchorPosition",
        "title_text_obj": "createTitleTextObject",

        "val_text_group": "createValueTextGroup",
        "val_text_shape": "calculateValueTextShape",
        "val_text_anchor_pos": "calculateValueTextAnchorPosition",
        
        "title_surf": "createTitleSurface",
        "val_text_surf": "createValueTextSurface",
        "static_bg_surf": "createStaticBackgroundSurface",
        "display_surf": "createDisplaySurface",
        
        "title_img_constructor": "createTitleImageConstructor",
        "val_text_img_constructor": "createValueTextImageConstructor",
        "static_bg_img_constructor": "createStaticBackgroundImageConstructor",
        "slider_img_constructor": "createSliderImageConstructor",
    }
    
    attribute_default_functions = {
        "slider_shape_rel": ((lambda obj: (0.7, 0.6)),),
        "slider_borders_rel": ((lambda obj: (0., 0.)),),
        #"title_text_group": ((lambda obj: SliderPlus.createTitleTextGroup()),),
        "title_anchor_type": ((lambda obj: "topleft"),),
        "title_text_color": ((lambda obj: text_color_def),),
        "val_text_anchor_type": ((lambda obj: "topleft"),),
        "val_text_color": ((lambda obj: text_color_def),),
        "val_text_dp": (sliderPlusDemarcationsDPDefault, ("slider",)),
    }
    
    fixed_attributes = set()
    
    sub_components = {
        "slider": {
            "class": Slider,
            "attribute_correspondence": {
                "shape": "slider_shape",
                "anchor_pos": "slider_bottomleft",
                "val_range": "val_range",
                "increment_start": "increment_start",
                "increment": "increment",
                "anchor_type": ((), lambda: "bottomleft"),
                "screen_topleft_offset": "topleft_screen",
                "init_val": "init_val",
                "demarc_numbers_text_group": "demarc_numbers_text_group",
                "demarc_numbers_dp": "demarc_numbers_dp",
                "thumb_radius_rel": "thumb_radius_rel",
                "demarc_line_lens_rel": "demarc_line_lens_rel",
                "demarc_intervals": "demarc_intervals",
                "demarc_start_val": "demarc_start_val",
                "demarc_numbers_max_height_rel": "demarc_numbers_max_height_rel",
                "track_color": "track_color",
                "thumb_color": "thumb_color",
                "demarc_numbers_color": "demarc_numbers_color",
                "demarc_line_colors": "demarc_line_colors",
                "thumb_outline_color": "thumb_outline_color",
                "mouse_enabled": "mouse_enabled",
                "name": "name",
            },
            #"creation_function": Slider,
            "creation_function_args": {
                "shape": None,
                "anchor_pos": None,
                "val_range": None,
                "increment_start": None,
                "increment": None,
                "anchor_type": None,
                "screen_topleft_offset": None,
                "init_val": None,
                "demarc_numbers_text_group": None,
                "demarc_numbers_dp": None,
                "thumb_radius_rel": None,
                "demarc_line_lens_rel": None,
                "demarc_intervals": None,
                "demarc_start_val": None,
                "demarc_numbers_max_height_rel": None,
                "track_color": None,
                "thumb_color": None,
                "demarc_numbers_color": None,
                "demarc_line_colors": None,
                "thumb_outline_color": None,
                "mouse_enabled": None,
                "name": None,
            },
            "container_attr_resets": {
                "display_surf": {"display_surf": True},
            },
            #"attr_reset_component_funcs": {},
            "container_attr_derivation": {
                "val": ["val"],
            }
        }
    }
    
    static_bg_components = ["title"]
    displ_component_attrs = ["static_bg", "slider", "val_text"]
    
    def __init__(self,
        title: str,
        shape: Tuple[Real],
        anchor_pos: Tuple[Real],
        val_range: Tuple[Real],
        increment_start: Real,
        increment: Optional[Real]=None,
        anchor_type: Optional[str]=None,
        screen_topleft_offset: Optional[Tuple[Real]]=None,
        init_val: Optional[Real]=None,
        demarc_numbers_text_group: Optional["TextGroup"]=None,
        demarc_numbers_dp: Optional[int]=None,
        thumb_radius_rel: Optional[Real]=None,
        demarc_line_lens_rel: Optional[Tuple[Real]]=None,
        demarc_intervals: Optional[Tuple[Real]]=None,
        demarc_start_val: Optional[Real]=None,
        demarc_numbers_max_height_rel: Optional[Real]=None,
        track_color: Optional[ColorOpacity]=None,
        thumb_color: Optional[ColorOpacity]=None,
        demarc_numbers_color: Optional[ColorOpacity]=None,
        demarc_line_colors: Optional[ColorOpacity]=None,
        thumb_outline_color: Optional[ColorOpacity]=None,
        mouse_enabled: Optional[bool]=None,
        
        slider_shape_rel: Optional[Tuple[Real]]=None,
        slider_borders_rel: Optional[Tuple[Real]]=None,
        title_text_group: Optional["TextGroup"]=None,
        title_anchor_type: Optional[str]=None,
        title_text_color: Optional[ColorOpacity]=None,
        val_text_group: Optional["TextGroup"]=None,
        val_text_anchor_type: Optional[str]=None,
        val_text_color: Optional[ColorOpacity]=None,
        val_text_dp: Optional[int]=None,
        
        name: Optional[str]=None,
        **kwargs,
    ) -> None:
        checkHiddenKwargs(type(self), kwargs)
        if name is None:
            SliderPlus.unnamed_count += 1
            name = f"slider plus {self.unnamed_count}"
        #self.name = name
        SliderPlus.sliderplus_names.add(name)
        #print(locals().keys())
        kwargs2 = self.initArgsManagement(locals(), kwargs=kwargs)#, rm_args=["name"])
        #print(kwargs2.keys())
        #print("init_val" in kwargs2)
        super().__init__(**kwargs2)
        #print("\nAfter initialisation")
        #print(self.__dict__.keys())
        #print("_init_val" in self.__dict__.keys())


        #super().__init__(**self.initArgsManagement(locals(), kwargs=kwargs, rm_args=["name", "init_val"]), slider_held=False, val_raw=init_val)

    #def _createSlider(
    #    self,
    #    func: Callable,
    #    attr_arg_dict: Dict[str, str],
    #) -> "Slider":
    #    kwargs = {attr: getattr(self, arg) for arg, attr in attr_arg_dict.items()}
    #    res = func(**kwargs)
    
    #def createSlider(self) -> "Slider":
    #    #print("Creating slider")
    #    return self.createSubComponent("slider")
    """
    def createComponent(self, component: str) -> Optional[Any]:
        #return self._createSlider(Slider, attr_arg_dict)
        cls = type(self)
        comp_dict = cls.__dict__.get("component_dict", cls.createComponentDictionary())
        if component not in comp_dict.keys(): return None
        component_creator_func, component_attr_dict = comp_dict[component]
        component_creator = component_creator_func()
        kwargs = {attr: (getattr(self, arg) if isinstance(arg, str) else arg[0]) for arg, attr in component_attr_dict.items()}
        return component_creator(**kwargs)
        
        return res
            shape=self.slider_shape,
            anchor_pos=self.slider_topleft,
            val_range=self.val_range,
            increment_start=self.increment_start,
            increment=self.increment,
            anchor_type="topleft",
            screen_topleft_offset=self.topleft_screen,
            init_val=self.init_val,
            demarc_numbers_text_group=self.demarc_numbers_text_group,
            demarc_numbers_dp=self.demarc_numbers_dp,
            thumb_radius_rel=self.thumb_radius_rel,
            demarc_line_lens_rel=self.demarc_line_lens_rel,
            demarc_intervals: Optional[Tuple[Real]]=None,
            demarc_start_val: Optional[Real]=None,
            demarc_numbers_max_height_rel: Optional[Real]=None,
            track_color: Optional[ColorOpacity]=None,
            thumb_color: Optional[ColorOpacity]=None,
            demarc_numbers_color: Optional[ColorOpacity]=None,
            demarc_line_colors: Optional[ColorOpacity]=None,
            thumb_outline_color: Optional[ColorOpacity]=None,
            mouse_enabled: Optional[bool]=None,
            name: Optional[str]=None,
        )
    """
    def calculateValue(self) -> Real:
        print("Using calculateValue()")
        print(f"slider value = {self.slider.val}")
        return self.slider.val

    @staticmethod
    def sliderShapeCalculator(slider_plus_shape: Tuple[int, int], slider_shape_rel: Tuple[float, float]) -> Tuple[int, int]:
        return tuple(math.floor(x * y) for x, y in zip(slider_plus_shape, slider_shape_rel)) 

    def calculateSliderShape(self) -> Tuple[int, int]:
        return self.sliderShapeCalculator(self.shape, self.slider_shape_rel)
    
    def calculateSliderBottomLeft(self) -> Tuple[int]:
        return (0, self.shape[1])
    
    def calculateSliderBorders(self) -> Tuple[int]:
        return tuple(round(x * y) for x, y in zip(self.shape, self.slider_borders_rel))

    def _setTitleTextObjectAttribute(self, attr: str, text_obj_attr: str) -> None:
        print(f"Using _setTitleTextObjectAttribute() to set title text object attribute {text_obj_attr}")
        title_text_obj = self.__dict__.get("_title_text_obj", None)
        if title_text_obj is None:
            print("title_text_obj is None")
            return
        #print("hi2")
        val = getattr(self, attr)
        print(f"setting title object attribute {text_obj_attr} to {val}")
        #print(attr, val)
        orig_val = getattr(title_text_obj, text_obj_attr, None)
        setattr(title_text_obj, text_obj_attr, val)
        chng_val = getattr(title_text_obj, text_obj_attr, None)
        #print(orig_val, chng_val)
        if orig_val != chng_val:
            self.title_surf = None
        return
    
    def calculateTitleShape(self) -> Tuple[int]:
        print("Using calculateTitleShape()")
        shape = self.shape
        slider_shape = self.slider_shape
        slider_borders = self.slider_borders
        print(f"overall shape = {shape}, slider_shape = {slider_shape}, slider_borders = {slider_borders}")
        res = (slider_shape[0], shape[1] - (slider_shape[1] + slider_borders[1]))
        #print(f"title shape = {res}")
        return res 
    
    def calculateTitleAnchorPosition(self) -> Tuple[int]:
        res = topLeftAnchorOffset(self.title_shape, self.title_anchor_type)
        #print(f"title anchor position = {res}")
        return res
    
    @staticmethod
    def createTitleTextGroup(
        font: Optional["pg.freetype"]=None,
        max_height: Optional[int]=None,
    ) -> "TextGroup":
        if font is None:
            font = font_def_func()
        return TextGroup(
            text_list=[],
            max_height0=max_height,
            font=None,
            font_size=None,
            min_lowercase=True,
            text_global_asc_desc_chars=None
        )
    
    def createTitleTextObject(self, title_text_group: Optional["TextGroup"]=None)\
            -> Union["Text", tuple]:
        print("creating title text object")
        txt = self.title
        if not txt: return ()
        if title_text_group is None:
            title_text_group = self.title_text_group
        color = self.title_text_color
        txt = self.title
        print(f"self.title_shape = {self.title_shape}")
        add_text_dicts = [{"text": txt, "font_color": color, "max_shape": self.title_shape, "_attr_reset_funcs": {"updated": [lambda obj, prev_val: setattr(obj, "title_surf", None)]}}]
        text_obj = title_text_group.addTextObjects(add_text_dicts)[0]
        #print(f"text_obj.max_shape_actual = {text_obj.max_shape_actual}")
        #text_obj.max_shape = self.title_shape
        return text_obj
    
    def setTitleShape(self, prev_val: Optional[Tuple[int, int]]) -> None:
        #print("setting title shape for text object")
        return self._setTitleTextObjectAttribute("title_shape", "max_shape")
    
    def createTitleSurface(self)\
            -> Union["pg.Surface", tuple]:
        #print("hello1")
        title_text_obj = self.title_text_obj
        #print(title_text_obj)
        if not title_text_obj: return ()
        #print("hello2")
        surf = pg.Surface(self.title_shape, pg.SRCALPHA)
        surf.set_alpha(255)
        surf.fill((0, 0, 255))
        #print(self.title_anchor_pos, self.title_anchor_type)
        title_text_obj.draw(surf, anchor_pos=self.title_anchor_pos, anchor_type=self.title_anchor_type)
        return surf
    
    def createTitleImageConstructor(self)\
            -> Callable[["pg.Surface"], None]:
        return lambda obj, surf: (None if obj.title_surf == () else surf.blit(obj.title_surf, (0, 0)))
    
    
    def createStaticBackgroundSurface(self)\
            -> Union["pg.Surface", tuple]:
        #print("creating static background surface")
        surf = pg.Surface(self.shape, pg.SRCALPHA)
        surf.set_alpha(255)
        #surf.fill((0, 255, 0))
        
        constructor_attrs = [f"{attr}_img_constructor"\
                for attr in self.static_bg_components]
        for constructor_attr in constructor_attrs:
            #print(f"adding {constructor_attr}")
            constructor_func =\
                    getattr(self, constructor_attr, (lambda obj, surf: None))
            constructor_func(self, surf)
        return surf
    
    def createStaticBackgroundImageConstructor(self)\
            -> Callable[["pg.Surface"], None]:
        return lambda obj, surf: (None if obj.static_bg_surf == () else surf.blit(obj.static_bg_surf, (0, 0)))
    
    def createSliderImageConstructor(self) -> Callable[["pg.Surface"], None]:
        #print("creating slider image constructor")
        return lambda obj, surf: obj.slider.draw(surf)


    def _setValueTextObjectAttribute(self, attr: str, text_obj_attr: str) -> None:
        #print("hi1")
        val_text_objs = self.__dict__.get("_val_text_objs", None)
        if val_text_objs is None: return
        #print("hi2")
        val = getattr(self, attr)
        #print(attr, val)
        orig_val = getattr(val_text_objs[0], text_obj_attr, None)
        it = [0] if attr == "text" else range(len(val_text_objs))
        for i in it:
            setattr(val_text_objs[i], text_obj_attr, val)
        chng_val = getattr(val_text_objs[0], text_obj_attr, None)
        #print(orig_val, chng_val)
        if orig_val != chng_val:
            self.val_text_surf = None
        return

    #@property
    #def val_text_group(self):
    #    res = getattr(self, "_val_text_group", None)
    #    if res is None:
    #        res = self.createValueTextGroup()
    #        self._val_text_group = res
    #    return res
    
    @property
    def val_text_objs(self):
        res = getattr(self, "_val_text_objs", None)
        #print(res)
        if res is None:
            res = self.createValueTextObjects()
            self._val_text_objs = res
        #print("hello")
        #print(res)
        return res
    
    def createValueTextGroup(self) -> TextGroup:
        return TextGroup([], max_height0=None, font=None,\
                font_size=None, min_lowercase=True)

    def createValueTextObjects(self) -> List[Text]:
        print("Using createValueTextObjects()")
        max_val = self.slider.val_range_actual[1]
        max_val_int = math.floor(max_val)
        max_int_n_dig = 0
        while max_val_int:
            max_val_int //= 10
            max_int_n_dig += 1
        max_int_n_dig = max(max_int_n_dig, 1)
        dp = self.val_text_dp
        #print(f"max_val_int = {max_val_int}, max_int_n_dig {max_int_n_dig}")
        def repeatedDigitString(d: int) -> str:
            l = str(d)
            res = l * max_int_n_dig if dp <= 0 else ".".join([l * max_int_n_dig, l * dp])
            #print(res)
            return res
            #return ".".join([l * max_int_n_dig, l * ])

        #nums = [str(d) * max_n_dig for d in range(10)]
        """
        val_max_width = self.arena_shape[0] * 0.1
        val_max_width_pixel = num_max_width * self.head_size
        num_anchor_pos = (self.border[0][0] + self.arena_shape[0] - num_max_width, self.border[1][0] * 0.9)
        num_anchor_pos_pixel = tuple(x * self.head_size for x in num_anchor_pos)
        txt_max_width = self.arena_shape[0] * 0.3
        txt_max_width_pixel = txt_max_width * self.head_size
        txt_anchor_pos = num_anchor_pos
        txt_anchor_pos_pixel = tuple(x * self.head_size for x in txt_anchor_pos)

        
        
        max_h = self.border[1][0] * 0.25
        max_h_pixel = max_h * self.head_size
        """
        #txt_shape = (10, 10)
        #add_text_dicts = [{"text": txt, "font_color": color, "max_shape": self.title_shape, "_attr_reset_funcs": {"updated": [lambda obj, prev_val: setattr(obj, "title_surf", None)]}}]
        text_dict = {"text": self.val_str, "anchor_pos0": self.val_text_anchor_pos, "anchor_type0": self.val_text_anchor_type, "max_shape": self.val_text_shape, "font_color": self.val_text_color, "_attr_reset_funcs": {"updated": [lambda obj, prev_val: setattr(obj, "val_text_surf", None)]}}
        text_list = []
        
        text_list.append(dict(text_dict))
        text_dict.pop("_attr_reset_funcs")
        for d in range(10):
            text_dict["text"] = repeatedDigitString(d)
            #print(text_dict["text"])
            text_list.append(dict(text_dict))
        grp = self.val_text_group
        text_objs = grp.addTextObjects(text_list)
        #print(text_objs)
        return text_objs

    def calculateValueString(self) -> str:
        print("Using calculateValueString()")
        dp = self.val_text_dp
        print(f"val_text_dp = {dp}")
        s = f"{self.val:.{dp}f}"
        print(f"value string = {s}")
        return f"{s}"

    @property
    def val_text_obj(self):
        return self.val_text_objs[0]

    def calculateValueTextShape(self) -> Tuple[int]:
        print("Using calculateValueTextShape()")
        shape = self.shape
        print(f"object shape = {shape}")
        slider_shape = self.slider_shape
        slider_borders = self.slider_borders
        res = (shape[0] - (slider_shape[0] + slider_borders[0]), shape[1])
        print(f"value shape = {res}")
        return res 
    
    def calculateValueTextAnchorPosition(self) -> Tuple[int]:
        slider_shape = self.slider_shape
        slider_borders = self.slider_borders
        res = tuple([slider_shape[0] + slider_borders[0], 0])
        print(f"value text anchor position = {res}")
        return res
    
    def setValueTextShape(self, prev_val: Optional[Tuple[int, int]]) -> None:
        #print("setting title shape for text object")
        return self._setValueTextObjectAttribute("val_text_shape", "max_shape")
    
    def createValueTextImageConstructor(self):
        #print("creating text image constructors")
        res = []
        def textImageConstructor(obj: SliderPlus, surf: "pg.Surface") -> None:
            obj.val_text_obj.font_color = obj.val_text_color
            obj.val_text_obj.text = obj.val_str
            obj.val_text_obj.max_shape = obj.val_text_shape
            print(f"value text shape = {obj.val_text_shape}, text object max text shape = {obj.val_text_obj.max_shape}")
            print(f"number of val_text_objs = {len(obj.val_text_objs)}")
            for i in range(1, len(obj.val_text_objs)):
                print(f"i = {i}, text = {obj.val_text_objs[i].text}")
                obj.val_text_objs[i].max_shape = obj.val_text_shape
            text_img = obj.val_text_surf
            if text_img == (): return
            surf.blit(text_img, obj.val_text_anchor_pos)
            #return lambda obj, surf: (None if obj.static_bg_surf == () else surf.blit(obj.static_bg_surf, (0, 0)))
            #obj.val_text_obj.draw(surf, anchor_pos=obj.val_text_anchor_pos, anchor_type=obj.val_text_anchor_type)
            #print("hello2")
            return
        return textImageConstructor

    #def createValueTextImageConstructor(self)\
    #        -> Callable[["pg.Surface"], None]:
    #    print("calling createValueTextImageConstructor()")
    #    return lambda obj, surf: (None if obj.val_text_surf == () else surf.blit(obj.val_text_surf, (0, 0)))

    def createValueTextSurface(self)\
            -> Union["pg.Surface", tuple]:
        print("Using createValueTextSurface()")
        #print("hello1")
        #print("about to get val_text_obj")
        val_text_obj = self.val_text_obj
        #print(val_text_obj)
        #print(title_text_obj)
        if not val_text_obj: return ()
        #print("hello2")
        surf = pg.Surface(self.val_text_shape, pg.SRCALPHA)
        #surf.set_alpha(100)
        surf.fill((0, 255, 255))
        print(self.title_anchor_pos, self.title_anchor_type)
        anchor_offset = topLeftAnchorOffset(self.val_text_shape, self.val_text_anchor_type)
        print(f"anchor offset = {anchor_offset}")
        val_text_obj.draw(surf, anchor_pos=anchor_offset, anchor_type=self.val_text_anchor_type)
        return surf
    
    """
    def createStaticBackgroundSurface(self)\
            -> Union["pg.Surface", tuple]:
        #print("creating static background surface")
        surf = pg.Surface(self.shape, pg.SRCALPHA)
        surf.set_alpha(255)
        surf.fill((0, 255, 0))
        
        constructor_attrs = [f"{attr}_img_constructor"\
                for attr in self.static_bg_components]
        for constructor_attr in constructor_attrs:
            #print(f"adding {constructor_attr}")
            constructor_func =\
                    getattr(self, constructor_attr, (lambda obj, surf: None))
            constructor_func(self, surf)
        return surf
    """
    def createDisplaySurface(self) -> Optional["pg.Surface"]:
        print("creating display surface")
        print(f"shape = {self.shape}")
        surf = pg.Surface(self.shape, pg.SRCALPHA)
        #surf.set_alpha(100)
        #surf.fill((255, 0, 0))
        print(self.displ_component_attrs)
        for attr in self.displ_component_attrs:
            print(f"{attr}_img_constructor")
            constructor_func = getattr(self, f"{attr}_img_constructor", (lambda obj, surf: None))
            #print(attr, surf)
            constructor_func(self, surf)
            #print("howdy")
        #print(f"display surface = {surf}")
        return surf
    
    def draw(self, surf: "pg.Surface") -> None:
        surf.blit(self.display_surf, self.topleft)
        return

    def eventLoop(self, events: Optional[List[int]]=None,\
            keys_down: Optional[List[int]]=None,\
            mouse_status: Optional[Tuple[int]]=None,\
            check_axes: Tuple[int]=(0, 1))\
            -> Tuple[bool, bool, bool, Any]:
        #print("calling SliderPlus eventLoop()")
        #print(events)
        ((quit, esc_pressed), (events, keys_down, mouse_status, check_axes)) = self.getEventLoopArguments(events=events, keys_down=keys_down, mouse_status=mouse_status, check_axes=check_axes)
        #print(events)
        running = not quit and not esc_pressed
        screen_changed = False
        
        (quit2, running2, screen_changed2, val_dict) = self.eventLoopComponents(
            events=events,
            keys_down=keys_down,
            mouse_status=mouse_status,
            check_axes=check_axes,
        )
        quit = quit or quit2
        running = running and running2
        screen_changed = screen_changed or screen_changed2
        
        #print()
        #print(quit, running, screen_changed, self.slider.val)
        return quit, running, screen_changed, self.val

class SliderPlusGroupElement(ComponentGroupElementBaseClass, SliderPlus):
    
    group_cls_func = lambda: SliderPlusGroup
    group_obj_attr = "slider_plus_group"
    #fixed_attributes = {group_obj_attr}

    #sub_components = dict(SliderPlus.sub_components)
    #sub_components["slider"]["class"] = SliderGroupElement

    sub_components = {
        "slider": {
            "class": SliderGroupElement,
            "attribute_correspondence": {
                "val_range": "val_range",
                "increment_start": "increment_start",
                "increment": "increment",
                "screen_topleft_offset": "topleft_screen",
                "init_val": "init_val",
                "demarc_numbers_dp": "demarc_numbers_dp",
                "demarc_intervals": "demarc_intervals",
                "demarc_start_val": "demarc_start_val",
                "name": "name",
            },
            "creation_function": (lambda slider_plus_group, **kwargs: slider_plus_group.slider_group.addSlider(**kwargs)),
            "creation_function_args": {
                "slider_plus_group": "slider_plus_group",
                #"shape": (("shape", "slider_shape_rel"), SliderPlus.sliderShapeCalculator),
                "anchor_pos": "slider_bottomleft",
                "val_range": None,
                "increment_start": None,
                "increment": None,
                "anchor_type": ((), (lambda: "bottomleft")),
                "screen_topleft_offset": None,
                "init_val": None,
                "demarc_numbers_dp": None,
                "demarc_intervals": None,
                "demarc_start_val": None,
                "name": None,
            },
            "container_attr_resets": {
                "display_surf": {"display_surf": True},
            },
            #"attr_reset_component_funcs": {},
            "container_attr_derivation": {
                "val": ["val"],
            }
        }
    }

    def __init__(
        self,
        slider_plus_group: "SliderPlusGroup",
        title: str,
        anchor_pos: Tuple[Real],
        val_range: Tuple[Real],
        increment_start: Real,
        increment: Optional[Real]=None,
        anchor_type: Optional[str]=None,
        screen_topleft_offset: Optional[Tuple[Real]]=None,
        init_val: Optional[Real]=None,
        demarc_numbers_dp: Optional[int]=None,
        demarc_intervals: Optional[Tuple[Real]]=None,
        demarc_start_val: Optional[Real]=None,
        
        val_text_dp: Optional[int]=None,
        
        name: Optional[str]=None,
        **kwargs,
    ) -> None:
        
        checkHiddenKwargs(type(self), kwargs)
        
        #self.__dict__[f"_{self.group_obj_attr}"] = slider_group
        super().__init__(
            shape=slider_plus_group.shape,
            title=title,
            anchor_pos=anchor_pos,
            val_range=val_range,
            increment_start=increment_start,
            increment=increment,
            anchor_type=anchor_type,
            screen_topleft_offset=screen_topleft_offset,
            init_val=init_val,
            demarc_numbers_text_group=slider_plus_group.demarc_numbers_text_group,
            demarc_numbers_dp=demarc_numbers_dp,
            thumb_radius_rel=slider_plus_group.thumb_radius_rel,
            demarc_line_lens_rel=slider_plus_group.demarc_line_lens_rel,
            demarc_intervals=demarc_intervals,
            demarc_start_val=demarc_start_val,
            demarc_numbers_max_height_rel=slider_plus_group.demarc_numbers_max_height_rel,
            track_color=slider_plus_group.track_color,
            thumb_color=slider_plus_group.thumb_color,
            demarc_numbers_color=slider_plus_group.demarc_numbers_color,
            demarc_line_colors=slider_plus_group.demarc_line_colors,
            thumb_outline_color=slider_plus_group.thumb_outline_color,
            mouse_enabled=slider_plus_group.mouse_enabled,
            name=name,
            _group=slider_plus_group,
            **kwargs,
        )

    def calculateSliderShape(self) -> Tuple[int, int]:
        return self.slider.shape

class SliderPlusGroup(ComponentGroupBaseClass):
    group_element_cls_func = lambda: SliderPlusGroupElement
    
    reset_graph_edges = {}
    
    """
    custom_reset_methods = {
        "slider_shape": "setSliderShape",
        "demarc_numbers_text_group": "setDemarcationNumbersTextGroup",
        "thumb_radius_rel": "setThumbRadiusRelative",
        "demarc_line_lens_rel": "setDemarcationLineLengthsRelative",
        "demarc_numbers_max_height_rel": "setDemarcationNumbersMaxHeightRelative",
        "track_color": "setTrackColor",
        "thumb_color": "setThumbColor",
        "demarc_numbers_color": "setDemarcationNumbersColor",
        "demarc_line_colors": "setDemarcationLineColors",
        "thumb_outline_color": "setThumbOutlineColor",
        "mouse_enabled": "setMouseEnabled",
    }
    """
    attribute_calculation_methods = {}
    
    # Review- account for using element_inherited_attributes in ComponentGroupBaseClass
    attribute_default_functions = {
        attr: SliderPlus.attribute_default_functions.get(attr) for attr in
        [
            "slider_shape_rel",
            "slider_borders_rel",
            "title_anchor_type",
            "title_text_color",
            "val_text_anchor_type",
            "val_text_color",
            "val_text_dp",
        ]
    }
    
    #fixed_attributes = {"sliders"}

    """
    slider_shape: Tuple[Real],
        demarc_numbers_text_group: Optional["TextGroup"]=None,
        thumb_radius_rel: Optional[Real]=None,
        demarc_line_lens_rel: Optional[Tuple[Real]]=None,
        demarc_numbers_max_height_rel: Optional[Real]=None,
        track_color: Optional[ColorOpacity]=None,
        thumb_color: Optional[ColorOpacity]=None,
        demarc_numbers_color: Optional[ColorOpacity]=None,
        demarc_line_colors: Optional[ColorOpacity]=None,
        thumb_outline_color: Optional[ColorOpacity]=None,
        mouse_enabled
    """

    sub_components = {
        "slider_group": {
            "class": SliderGroup,
            "attribute_correspondence": {
                "slider_shape": (("shape", "slider_shape_rel"), SliderPlus.sliderShapeCalculator),
                "demarc_numbers_text_group": "demarc_numbers_text_group",
                "thumb_radius_rel": "thumb_radius_rel",
                "demarc_line_lens_rel": "demarc_line_lens_rel",
                "demarc_numbers_max_height_rel": "demarc_numbers_max_height_rel",
                "track_color": "track_color",
                "thumb_color": "thumb_color",
                "demarc_numbers_color": "demarc_numbers_color",
                "demarc_line_colors": "demarc_line_colors",
                "thumb_outline_color": "thumb_outline_color",
                "mouse_enabled": "mouse_enabled",
            },
            #"creation_function": SliderGroup,
            "creation_function_args": {
                "slider_shape": (("shape", "slider_shape_rel"), SliderPlus.sliderShapeCalculator),
                "demarc_numbers_text_group": None,
                "thumb_radius_rel": None,
                "demarc_line_lens_rel": None,
                "demarc_numbers_max_height_rel": None,
                "track_color": None,
                "thumb_color": None,
                "demarc_numbers_color": None,
                "demarc_line_colors": None,
                "thumb_outline_color": None,
                "mouse_enabled": None,
            },
        }
    }
    
    element_inherited_attributes = {
        "shape": "shape",
        "demarc_numbers_text_group": "demarc_numbers_text_group",
        "thumb_radius_rel": "thumb_radius_rel",
        "demarc_line_lens_rel": "demarc_line_lens_rel",
        "demarc_numbers_max_height_rel": "demarc_numbers_max_height_rel",
        "track_color": "track_color",
        "demarc_numbers_color": "demarc_numbers_color",
        "demarc_line_colors": "demarc_line_colors",
        "thumb_outline_color": "thumb_outline_color",
        "mouse_enabled": "mouse_enabled",
        "title_text_group": "title_text_group",
        "title_anchor_type": "title_anchor_type",
        "title_text_color": "title_text_color",
        "val_text_group": "val_text_group",
        "val_text_anchor_type": "val_text_anchor_type",
        "val_text_color": "val_text_color",
    }

    def __init__(self, 
        shape: Tuple[Real],
        demarc_numbers_text_group: Optional["TextGroup"]=None,
        thumb_radius_rel: Optional[Real]=None,
        demarc_line_lens_rel: Optional[Tuple[Real]]=None,
        demarc_intervals: Optional[Tuple[Real]]=None,
        demarc_start_val: Optional[Real]=None,
        demarc_numbers_max_height_rel: Optional[Real]=None,
        track_color: Optional[ColorOpacity]=None,
        thumb_color: Optional[ColorOpacity]=None,
        demarc_numbers_color: Optional[ColorOpacity]=None,
        demarc_line_colors: Optional[ColorOpacity]=None,
        thumb_outline_color: Optional[ColorOpacity]=None,
        mouse_enabled: Optional[bool]=None,
        slider_shape_rel: Optional[Tuple[Real]]=None,
        slider_borders_rel: Optional[Tuple[Real]]=None,
        title_text_group: Optional["TextGroup"]=None,
        title_anchor_type: Optional[str]=None,
        title_text_color: Optional[ColorOpacity]=None,
        val_text_group: Optional["TextGroup"]=None,
        val_text_anchor_type: Optional[str]=None,
        val_text_color: Optional[ColorOpacity]=None,
        **kwargs,
    ) -> None:
        checkHiddenKwargs(type(self), kwargs)

        super().__init__(**self.initArgsManagement(locals(), kwargs=kwargs))

    def addSliderPlus(
        self,
        title: str,
        anchor_pos: Tuple[Real],
        val_range: Tuple[Real],
        increment_start: Real,
        increment: Optional[Real]=None,
        anchor_type: Optional[str]=None,
        screen_topleft_offset: Optional[Tuple[Real]]=None,
        init_val: Optional[Real]=None,
        demarc_numbers_dp: Optional[int]=None,
        demarc_intervals: Optional[Tuple[Real]]=None,
        demarc_start_val: Optional[Real]=None,
        val_text_dp: Optional[int]=None,
        name: Optional[str]=None,
        **kwargs,
    ) -> "SliderGroupElement":
        
        res = self._addElement(
            title=title,
            anchor_pos=anchor_pos,
            val_range=val_range,
            increment_start=increment_start,
            increment=increment,
            anchor_type=anchor_type,
            screen_topleft_offset=screen_topleft_offset,
            init_val=init_val,
            demarc_numbers_dp=demarc_numbers_dp,
            demarc_intervals=demarc_intervals,
            demarc_start_val=demarc_start_val,
            val_text_dp=val_text_dp,
            name=name,
        )
        return res

class SliderPlusVerticalBattery:
    def __init__(self, screen, x, y, w, h, slider_gap_rel=0.2, track_color=None,
            thumb_color=None, thumb_radius_rel=1, font=None,
            demarc_line_lens_rel=None, number_size_rel=2,
            numbers_color=None, demarc_line_colors=None,
            thumb_outline_color=None, slider_w_prop=0.75,
            title_size_rel=0.4, title_color=None, title_gap_rel=0.2,
            val_text_size_rel=0.4, val_text_color=None):
        self.screen = screen
        self.dims = (x, y, w, h)
        self.slider_gap_rel = slider_gap_rel
        self.track_color = track_color_def if track_color is None else\
            track_color
        self.thumb_radius_rel = thumb_radius_rel
        self.font = font_def_func() if font is None else font
        self.demarc_line_lens_rel = demarc_line_lens_rel
        self.number_size_rel = number_size_rel
        self.slider_w_prop = slider_w_prop
        self.title_size_rel = title_size_rel
        self.title_gap_rel = title_gap_rel
        self.val_text_size_rel = val_text_size_rel

        self.track_color = track_color_def if track_color is None else\
            track_color
        self.numbers_color = text_color_def if numbers_color is None\
            else numbers_color
        if demarc_line_colors is None:
            self.demarc_line_colors = (self.track_color,)
        elif isinstance(demarc_line_colors[0], int):
            self.demarc_line_colors = (tuple(demarc_line_colors),)
        else: self.demarc_line_colors = tuple(tuple(c) for c in demarc_line_colors)
        self.thumb_color = thumb_color_def if thumb_color is None\
                else thumb_color
        self.thumb_outline_color = thumb_outline_color
        self.title_color = text_color_def if title_color is None\
                else title_color
        self.val_text_color = self.title_color if val_text_color is None\
                else val_text_color
         
        self.sliderPlus_objects = []
        self.sliderPlus_dims_set = True
        self.vals = []

    def addSliderPlus(self, title, val_range=(0, 100), demarc_intervals=(20, 10, 5),\
            demarc_start_val=0, increment=None, increment_start=None, default_val=None, numbers_dp=0):
        self.sliderPlus_objects.append(
            SliderPlus(title, self.screen, *self.dims, val_range=val_range,
                demarc_intervals=demarc_intervals,
                demarc_start_val=demarc_start_val,
                increment=increment, increment_start=increment_start, track_color=self.track_color,
                thumb_color=self.thumb_color, thumb_radius_rel=self.thumb_radius_rel,
                default_val=default_val, font=self.font,
                demarc_line_lens_rel=self.demarc_line_lens_rel,
                numbers_dp=numbers_dp, number_size_rel=self.number_size_rel,
                numbers_color=self.numbers_color, demarc_line_colors=self.demarc_line_colors,
                thumb_outline_color=self.thumb_outline_color,
                slider_w_prop=self.slider_w_prop,
                title_size_rel=self.title_size_rel,
                title_color=self.title_color, title_gap_rel=self.title_gap_rel,
                val_text_size_rel=self.val_text_size_rel,
                val_text_color=self.val_text_color
            )
        )
        self.sliderPlus_dims_set = False
        self.vals.append(self.sliderPlus_objects[-1].val)
    
    @property
    def dims(self):
        return self._dims
    
    @dims.setter
    def dims(self, dims):
        self._sliderPlus_dims = None
        self._dims = dims
        return
    
    @property
    def sliderPlus_dims(self):
        res = getattr(self, "_sliderPlus_dims", None)
        if res is None:
            y_lst, h = self.findSliderPlusVerticalDimensions()
            res = [(self.dims[0], y, self.dims[2], h) for y in y_lst]
            self._sliderPlus_dims = res
        return res
    
    def findSliderPlusVerticalDimensions(self):
        n = len(self.sliderPlus_objects)
        h_plus_gap = self.dims[3] / (n - self.slider_gap_rel)
        h = math.floor((1 - self.slider_gap_rel) * h_plus_gap)
        y_lst = [self.dims[1] + math.floor(i * h_plus_gap) for i in range(n)]
        return y_lst, h
    
    def setSliderPlusDimensions(self):
        if self.sliderPlus_dims_set: return
        for slider_plus, dims in zip(self.sliderPlus_objects, self.sliderPlus_dims):
            slider_plus.dims = dims
        self.sliderPlus_dims_set = True
        return
    
    def event_loop(self, mouse_pos=None, keys_pressed=None,
            mouse_down=None, mouse_clicked=False,
            nav_keys=None, nav_keys_active=False):
        screen_changed = False
        for i, slider_plus in enumerate(self.sliderPlus_objects):
            change, val = slider_plus.event_loop(\
                    mouse_pos=mouse_pos, keys_pressed=keys_pressed,\
                    mouse_down=mouse_down, mouse_clicked=mouse_clicked,\
                    nav_keys=nav_keys, nav_keys_active=nav_keys_active)
            if change:
                screen_changed = True
                self.vals[i] = val
        return screen_changed, self.vals
    
    def draw(self):
        if not self.sliderPlus_dims_set:
            self.setSliderPlusDimensions()
        for slider_plus in self.sliderPlus_objects:
            slider_plus.draw()
        pygame.draw.rect(self.screen, (0, 100, 255), self.dims, 1)
        return

        
        
        
        
