#!/usr/bin/env python

import bisect
import copy
import functools
import heapq
import os
import sys
import weakref

from abc import ABC, abstractmethod
from collections import deque

from typing import Union, Tuple, List, Set, Dict, Optional, Callable, Any

import pygame as pg

from .config import (
    enter_keys_def_glob,
    navkeys_def_glob,
    mouse_lclicks,
    named_colors_def,
    font_def_func,
)
from .utils import Real

from .user_input_processing import (
    UserInputProcessor,
    createNavkeyDict,
)
from .position_offset_calculators import topLeftFromAnchorPosition

def checkHiddenKwargs(cls: type, _kwargs: Dict[str, Any]) -> None:
    for arg in _kwargs.keys():
        if not arg.startswith("_"):
            raise ValueError(
                f"{cls.__name__}.__init__() got an unexpected "
                f"keyword argument '{arg}'"
            )
    return

def findSmallestAvailableIndex(
    lst: List[Any],
    available_idx_heap: List[int]
) -> int:
    idx = None
    while available_idx_heap:
        idx = heapq.heappop(available_idx_heap)
        if idx >= len(lst):
            for _ in range(len(available_idx_heap)):
                available_idx_heap.pop()
            break
        if lst[idx] is None:
            break
    else:
        idx = len(lst)
        lst.append(None)
    return idx

def popIndex(
    idx: int,
    lst: List[Any],
    available_idx_heap: List[int]
) -> Any:
    res = lst[idx]
    lst[idx] = None
    if idx < len(lst) - 1:
        heapq.heappush(available_idx_heap, idx)
        return res
    lst.pop()
    while lst and lst[-1] is None:
        lst.pop()
    if available_idx_heap and len(lst) <= available_idx_heap[0]:
        for _ in range(len(available_idx_heap)):
            available_idx_heap.pop()
    return res
    

# Review:
# Consider having custom reset functions for instances, in particular
# to raise errors when trying to reset an attribute whose value is
# inherited from its container

class ComponentBaseClass(ABC):

    finalizer_attributes = {"name"}
    
    def __init__(self, **kwargs):
        #print("Initialising ComponentBaseClass")
        #print(kwargs)
        #print("init_val" in kwargs)
        #print("hello")
        #print(type(self).__name__)
        #print(kwargs)
        #print(f"_from_group = {kwargs.get('_from_group', False)}")
        #b = "_attr_reset_funcs" in kwargs.keys():
        #if b:
        #    print("hello")
        #    print(kwargs)
        self.setAttributes(self.initArgsManagement(locals(), kwargs=kwargs, rm_args=["_from_group"]), _from_group=kwargs.get("_from_group", False), _from_container=(kwargs.get("_container_obj", None) is not None))
        #if b:
        #    print("hello2")
        #cls = type(self)
        self.weakref_finalizer
    
    @classmethod
    def _addInstanceToRecord(cls, obj: "ComponentGroupElementBaseClass", finalizer_attr_vals: Dict[str, Any]) -> int:
        if "instances" not in cls.__dict__.keys():
            cls.instances = []
            cls.instances_idx_dict = {}
            cls.available_inst_idx_heap = []
            cls.instances_finalizer_attr_vals = []
        
        
        #print("\nAdding instance to record")
        #print(f"cls = {cls.__name__}")
        #print(f"cls.instances_finalizer_attr_vals before = {cls.instances_finalizer_attr_vals}")
        #print(f"cls.instances = {cls.instances}")
        #print(f"cls.available_inst_idx_heap = {cls.available_inst_idx_heap}")
        
        w_ref = weakref.ref(obj)
        if w_ref in cls.instances_idx_dict.keys():
            return cls.instances_idx_dict[w_ref]
        idx = findSmallestAvailableIndex(cls.instances, cls.available_inst_idx_heap)
        
        cls.instances[idx] = w_ref
        cls.instances_idx_dict[w_ref] = idx
        if idx == len(cls.instances_finalizer_attr_vals):
            cls.instances_finalizer_attr_vals.append(None)
        cls.instances_finalizer_attr_vals[idx] = finalizer_attr_vals
        obj._inst_idx = idx
        #print(f"idx = {idx}")
        #print(f"cls.instances_finalizer_attr_vals after = {cls.instances_finalizer_attr_vals}")
        #print(f"cls.instances = {cls.instances}")
        #print(f"cls.available_inst_idx_heap = {cls.available_inst_idx_heap}")
        return idx
    
    @classmethod
    def _removeInstanceFromRecord(cls, inst_idx: int) -> None:
        
        #print("\nRemoving instance from record")
        #print(f"cls = {cls.__name__}")
        #print(f"inst_idx = {inst_idx}")
        #print(f"cls.instances_finalizer_attr_vals before = {cls.instances_finalizer_attr_vals}")
        #print(f"cls.instances = {cls.instances}")
        #print(f"cls.available_inst_idx_heap = {cls.available_inst_idx_heap}")
        w_ref = popIndex(
            inst_idx,
            cls.instances,
            cls.available_inst_idx_heap
        )
        
        cls.instances_idx_dict.pop(w_ref)
        cls.instances_finalizer_attr_vals[inst_idx] = None
        for _ in range(len(cls.instances_finalizer_attr_vals) - len(cls.instances)):
            cls.instances_finalizer_attr_vals.pop()
        #print(f"cls.instances_finalizer_attr_vals after = {cls.instances_finalizer_attr_vals}")
        #print(f"cls.instances = {cls.instances}")
        #print(f"cls.available_inst_idx_heap = {cls.available_inst_idx_heap}")
        return
    
    @classmethod
    def createFinalizerAttributeSet(cls) -> Set[str]:
        finalizer_attr_set = set()
        for cls2 in cls.mro():
            for attr in cls2.__dict__.get("finalizer_attributes", set()):
                finalizer_attr_set.add(attr if isinstance(attr, str) else attr(cls))
            #finalizer_attr_set |= cls2.__dict__.get("finalizer_attributes", set())
        cls.finalizer_attr_set = finalizer_attr_set
        return finalizer_attr_set
    
    @property
    def weakref_finalizer(self):
        res = getattr(self, "_weakref_finalizer", None)
        if res is None:
            res = self.createWeakrefFinalizer()
            self._weakref_finalizer = res
        return res
    
    def createWeakrefFinalizer(self):
        #print("creating weakref finalizer")
        cls = type(self)
        finalizer_attr_set = getattr(cls, "finalizer_attr_set", cls.createFinalizerAttributeSet())
        finalizer_attr_vals = {attr: getattr(self, attr, None) for attr in finalizer_attr_set}
        inst_idx = cls._addInstanceToRecord(self, finalizer_attr_vals)
        #print(cls.finalizer_kwargs, kwargs)
        return weakref.finalize(self, cls.remove, inst_idx)
    
    @classmethod
    def remove(cls, inst_idx: int) -> None:
        attrs = cls.instances_finalizer_attr_vals[inst_idx]
        #nm = attrs.get("name", None)
        #if nm is not None:
        #    print(f"Removed {cls.__name__} object {nm}")
        #else: print(f"Removed {cls.__name__}")
        
        cls._removeInstanceFromRecord(inst_idx)
        return
    
    def initArgsManagement(self, init_locals: Dict[str, Any], kwargs: Optional[Dict[str, Any]]=None, rm_args: Optional[List[str]]=None) -> Dict[str, Any]:
        res = dict(init_locals)
        res.pop("self")
        #print("init_val" in res.keys())
        if "__class__" in res:
            res.pop("__class__")
        if kwargs is not None:
            res.pop("kwargs")
            for k, v in kwargs.items():
                res[k] = v
        if rm_args is not None:
            for arg in rm_args:
                if arg in res.keys():
                    res.pop(arg)
        return res
    
    @staticmethod
    def _attr2Index(
        attr: str,
        attr_list: List[str],
        attr_dict: Dict[str, int],
        reset_graph: List[Dict[int, Union[List[Callable[["DisplayComponentBase"], bool]]]]]
    ) -> int:
        if attr in attr_dict.keys():
            return attr_dict[attr]
        idx = len(attr_list)
        attr_dict[attr] = idx
        attr_list.append(attr)
        reset_graph.append({})
        return idx 
    
    @classmethod
    def createResetStructures(cls) -> Tuple[Union[List[str], Dict[str, int], List[Dict[int, Union[List[Callable[["DisplayComponentBase"], bool]]]]], Dict[int, List[Callable[[], None]]]]]:
        attr_list = []
        attr_dict = {}
        reset_graph = []
        custom_reset_funcs = {}
        
        def attr2Index(attr: str) -> int:
            return cls._attr2Index(attr, attr_list, attr_dict, reset_graph)
        
        def addEdgeFunction(
            idx1: int,
            idx2: int,
            func: Union[List[Callable[["DisplayComponentBase"], bool]]],
            reset_graph: List[Dict[int, Union[List[Callable[["DisplayComponentBase"], bool]]]]]=reset_graph
        ) -> None:
            if func is True or reset_graph[idx1].get(idx2, False) is True:
                reset_graph[idx1][idx2] = True
            else:
                reset_graph[idx1].setdefault(idx2, [])
                reset_graph[idx1][idx2].append(func)
            return
        
        
        
        # Ensuring vertices edges in graph due to default dependencies
        # are included
        attr_deffunc_dict = cls.getAttributeDefaultFunctionDictionary()
        
        for attr1, (_, attr2_lst) in attr_deffunc_dict.items():
            idx1 = attr2Index(attr1)
            for attr2 in attr2_lst:
                #print(attr1, attr2)
                idx2 = attr2Index(attr2)
                addEdgeFunction(idx2, idx1, functools.partial(lambda attr, obj: attr in obj.__dict__.setdefault("is_default_set", set()), attr1), reset_graph=reset_graph)
        
        for cls2 in cls.mro():
            for attr1, attr2_dict in getattr(cls2, "reset_graph_edges", {}).items():
                #if attr1 == "val":
                #    print(attr1, attr2_dict)
                idx1 = attr2Index(attr1)
                for attr2, func_new in attr2_dict.items():
                    idx2 = attr2Index(attr2)
                    addEdgeFunction(idx1, idx2, func_new, reset_graph=reset_graph)
                    #if func_new is True or reset_graph[idx1].get(idx2, False) is True:
                    #    reset_graph[idx1][idx2] = True
                    #else:
                    #    reset_graph[idx1].setdefault(idx2, [])
                    #    reset_graph[idx1][idx2].append(func_new)
            for attr, method_name in cls2.__dict__.get("custom_reset_methods", {}).items():
                meth = cls2.__dict__.get(method_name, None)
                if meth is None:
                    continue
                idx = attr2Index(attr)
                custom_reset_funcs.setdefault(idx, [])
                custom_reset_funcs[idx].append(functools.partial(meth))
        
        def setComponentAttribute(component_name: str, component_attr_to_set: str, attr_calc: Union[str, Tuple[tuple, Callable]], obj: "ComponentBaseClass", prev_val: Any) -> None:
            component = obj.__dict__.get(f"_{component_name}", None)
            if component is None: return
            val = getattr(obj, attr_calc) if isinstance(attr_calc, str) else attr_calc[1](*[getattr(obj, x) for x in attr_calc[0]])
            return component.setAttributes({component_attr_to_set: val}, _from_container=True)
        
        sub_components_dict = cls.__dict__.get("sub_components_dict", cls._createSubComponentDictionary())
        for component_name, sc_dict in sub_components_dict.items():
            attr_dict2 = sc_dict["attribute_correspondence"]
            for component_attr, attr_calc in attr_dict2.items():
                parent_attr_lst = [attr_calc] if isinstance(attr_calc, str) else attr_calc[0]
                for parent_attr in parent_attr_lst:
                    
                    idx = attr2Index(parent_attr)
                    #print(parent_attr, idx)
                    custom_reset_funcs.setdefault(idx, [])
                    custom_reset_funcs[idx].append(functools.partial(setComponentAttribute, component_name, component_attr, attr_calc))
                #if not isinstance(val_attr, str): continue
                #for parent_attr in attr_lst:
                #    idx = attr2Index(parent_attr)
                #    custom_reset_funcs.setdefault(idx, [])
                #    custom_reset_funcs[idx].append(functools.partial(setComponentAttribute, component_name, component_attr, parent_attr))
        
        def updateFinalizerAttributeValue(attr: str, obj: "ComponentBaseClass", prev_val: Any) -> None:
            cls = type(obj)
            inst_idx = obj.__dict__.get("_inst_idx", None)
            if inst_idx is None: return
            cls.instances_finalizer_attr_vals[inst_idx][attr] = getattr(obj, attr)
            return
        
        #if "instances_finalizer_attr_vals" in cls.__dict__.keys():
        finalizer_attr_set = cls.__dict__.get("finalizer_attr_set", cls.createFinalizerAttributeSet())
        for attr in finalizer_attr_set:
            idx = attr2Index(attr)
            custom_reset_funcs.setdefault(idx, [])
            custom_reset_funcs[idx].append(functools.partial(updateFinalizerAttributeValue, attr))
        
        cls.reset_graph = reset_graph
        cls.attr_list = attr_list
        cls.attr_dict = attr_dict
        cls.custom_reset_funcs = custom_reset_funcs
        return (attr_list, attr_dict, reset_graph, custom_reset_funcs)
    
    def getComponentAttribute(self, component_name: str, component_attr: str) -> Any:
        return getattr(getattr(self, component_name), component_attr)
    
    @classmethod
    def _createAttributeCalculationFunctionDictionary(cls) -> Dict[str, Any]:
        attr_calcfunc_dict = {}
        is_calc_and_set = lambda method_name: method_name.startswith("calculateAndSet")
        
        def attributeCalculator(method_name: str, obj: "DisplayComponentBase", cls: type=cls) -> Any:
            return getattr(cls, method_name)(obj)
        
        for cls2 in cls.mro():
            meth_dict = cls2.__dict__.get("attribute_calculation_methods", {})
            for attr, method_name in meth_dict.items():
                if attr in attr_calcfunc_dict.keys():
                    continue
                elif not hasattr(cls, method_name):
                    raise AttributeError(f"The class '{cls.__name__}' has no method {method_name}(), which is required to calculate the value of attribute '{attr}' of its instances.")
                attr_calcfunc_dict[attr] = (functools.partial(attributeCalculator, method_name), is_calc_and_set(method_name))
        
        sub_components_dict = cls.__dict__.get("sub_components_dict", cls._createSubComponentDictionary())
        for component_nm, component_dict in sub_components_dict.items():
            if component_nm in attr_calcfunc_dict.keys(): continue
            attr_calcfunc_dict[component_nm] = (functools.partial((lambda cn, obj: obj.createSubComponent(cn)), component_nm), False)
            for component_attr, attrs in component_dict.get("container_attr_derivation", {}).items():
                for attr in attrs:
                    if attr in attr_calcfunc_dict.keys(): continue
                    if attr == "val": print(f"adding slider attribute {component_attr} to slider plus attribute {attr} in attr_calcfunc_dict")
                    attr_calcfunc_dict[attr] = (functools.partial((lambda cn, ca, obj: obj.getComponentAttribute(cn, ca)), component_nm, component_attr), False)
        return attr_calcfunc_dict

    @classmethod
    def getAttributeCalculationFunctionDictionary(cls) -> Dict[str, Any]:
        #print(f"Using getAttributeCalculationFunctionDictionary() for class {cls.__name__}")
        #if "_attr_calcfunc_dict" not in cls.__dict__.keys():
        if cls.__dict__.get("_attr_calcfunc_dict", None) is None:
            cls._attr_calcfunc_dict = cls._createAttributeCalculationFunctionDictionary()
            #print(f"set attr_calcfunc_dict for class {cls.__name__}")
        return cls._attr_calcfunc_dict

    @classmethod
    def _createAttributeDefaultFunctionDictionary(cls) -> Dict[str, Any]:
        #print(f"Using _createAttributeCalculationFunctionDictionary() for class {cls.__name__}")
        attr_deffunc_dict = {}
        for cls2 in cls.mro():
            for attr, def_func_tup in cls2.__dict__.get("attribute_default_functions", {}).items():
                #print(attr, def_func_tup)
                if callable(def_func_tup):
                    def_func_tup = (def_func_tup, ())
                elif len(def_func_tup) == 1:
                    def_func_tup = (def_func_tup[0], ())
                attr_deffunc_dict.setdefault(attr, def_func_tup)
        return attr_deffunc_dict
    
    @classmethod
    def getAttributeDefaultFunctionDictionary(cls) -> Dict[str, Any]:
        #if "_attr_deffunc_dict" not in cls.__dict__.keys():
        if cls.__dict__.get("_attr_deffunc_dict", None) is None:
            cls._attr_deffunc_dict = cls._createAttributeDefaultFunctionDictionary()
        return cls._attr_deffunc_dict
    
    @classmethod
    def _createFixedAttributeSet(cls) -> Set[str]:
        #print(f"Using _createFixedAttributeSet() for class {cls.__name__}")
        fixed_attr_set = set()
        for cls2 in cls.mro():
            fixed_attr_set |= cls2.__dict__.get("fixed_attributes", set())
        return fixed_attr_set

    @classmethod
    def getFixedAttributeSet(cls) -> Set[str]:
        #print(f"Using getFixedAttributeSet() for class {cls.__name__}")
        if cls.__dict__.get("_fixed_attr_set", None) is None:
            cls._fixed_attr_set = cls._createFixedAttributeSet()
        return cls._fixed_attr_set
    
    @classmethod
    def _createAttributeProcessingDictionary(cls) -> Dict[str, Callable[[Any], Any]]:
        #print(f"Using _createAttributeProcessingDictionary() for class {cls.__name__}")
        attr_processing_dict = {}
        for cls2 in cls.mro():
            for attr, process_func in cls2.__dict__.get("attribute_processing", {}).items():
                attr_processing_dict.setdefault(attr, process_func)
        return attr_processing_dict

    @classmethod
    def getAttributeProcessingDictionary(cls) -> Dict[str, Callable[[Any], Any]]:
        #if "_attr_processing_dict" not in cls.__dict__.keys():
        if cls.__dict__.get("_attr_processing_dict", None) is None:
            cls._attr_processing_dict = cls._createAttributeProcessingDictionary()
        return cls._attr_processing_dict
    
    @classmethod
    def processSubComponents(cls) -> Dict[str, Dict[str, Any]]:
        sub_components_processed = {}
        sub_components = cls.__dict__.get("sub_components", {})
        for sc_nm, sc_dict in sub_components.items():
            component_cls = sc_dict["class"]
            attr_corresp_dict = sc_dict.get("attribute_correspondence", {})
            creation_function = sc_dict.get("creation_function", component_cls)
            creation_function_args = sc_dict.get("creation_function_args", {})
            container_attr_resets = sc_dict.get("container_attr_resets", {})
            container_attr_derivation = sc_dict.get("container_attr_derivation", {})
            
            container_attr_resets2 = dict(container_attr_resets)
            for component_attr, container_attrs in container_attr_derivation.items():
                container_attr_resets2.setdefault(component_attr, {})
                for container_attr in container_attrs:
                    container_attr_resets2[component_attr][container_attr] = True
            #if container_attr_resets2:
            #    print(f"container_attr_resets2 = {container_attr_resets2}")
            sub_components_processed[sc_nm] = {
                "class": component_cls,
                "attribute_correspondence": attr_corresp_dict,
                "creation_function": creation_function,
                "creation_function_args": creation_function_args,
                "container_attr_resets": container_attr_resets2,
                "container_attr_derivation": container_attr_derivation,
            }
            
        cls.sub_components_processed = sub_components_processed
        return sub_components_processed
    
    # Review- consider adding in option for custom functions enabling
    # changes to the container object in response to changes to
    # the component (key "attr_reset_component_funcs")
    
    @classmethod
    def _createSubComponentDictionary(cls) -> Dict[str, Dict[str, Any]]:
        sub_components_dict = {}
        for cls2 in cls.mro():
            sub_components_processed = cls2.__dict__.get("sub_components_processed", cls.processSubComponents())
            for component, component_dict in sub_components_processed.items():
                sub_components_dict.setdefault(component, component_dict)
        cls.sub_components_dict = sub_components_dict
        return sub_components_dict
    
    @classmethod
    def createSubComponentDictionary(cls) -> Dict[str, Dict[str, str]]:
        cls.createResetStructures()
        return cls.sub_components_dict
    
    
    
    def createSubComponent(self, component: str) -> Optional[Any]:
        #return self._createSlider(Slider, attr_arg_dict)
        cls = type(self)
        sub_components_dict = cls.__dict__.get("sub_components_dict", cls.createSubComponentDictionary())
        if component not in sub_components_dict.keys(): return None
        #component_cls, component_attr_dict, component_creator, component_creator_args_dict, container_attr_reset_dict = sub_components_dict[component]
        sc_dict = sub_components_dict[component]
        component_cls = sc_dict["class"]
        attr_corresp_dict = sc_dict["attribute_correspondence"]
        creation_function = sc_dict["creation_function"]
        creation_function_args = sc_dict["creation_function_args"]
        container_attr_resets = sc_dict["container_attr_resets"]
        container_attr_derivation = sc_dict["container_attr_derivation"]
        
        #container_attr_resets2 = dict(container_attr_resets)
        #for component_attr, container_attrs in container_attr_derivation.items():
        #    container_attr_resets2.setdefault(component_attr, {})
        #    for container_attr in container_attrs:
        #        container_attr_resets2[component_attr][container_attr] = True
        #print("hello")
        #print(container_attr_resets2)
        kwargs = {}
        for arg, val_repr in creation_function_args.items():
            if val_repr is None:
                # Consider adding specific error in case arg not in attr_corresp_dict
                val_repr = attr_corresp_dict.get(arg, None)
                #if val_repr is None: continue
            #print(arg, val_repr, len(val_repr))
            kwargs[arg] = getattr(self, val_repr) if isinstance(val_repr, str) else val_repr[1](*[getattr(self, x) for x in val_repr[0]])
            #kwargs[arg] = getattr(self, val_repr) if isinstance(val_repr, str) else val_repr[0]
        kwargs["_container_obj"] = self
        if container_attr_resets:
            kwargs["_container_attr_reset_dict"] = container_attr_resets
        #print(f"subcomponent creation kwargs = {kwargs}")
        
        component = creation_function(**kwargs)
        setattr_dict = {arg: (getattr(self, attr) if isinstance(attr, str) else attr[1](*[getattr(self, x) for x in attr[0]])) for arg, attr in attr_corresp_dict.items()}
        component.setAttributes(setattr_dict)
        component._attr_container_dependent_set = {x for x in attr_corresp_dict.values() if isinstance(x, str)}
        return component
    
    def calculateAndSetAttribute(self, attr: str) -> bool:
        #print(f"calculating and setting attribute {attr} for object {self.__str__()}")
        cls = type(self)
        attr_calcfunc_dict = cls.getAttributeCalculationFunctionDictionary()
        #attr_calcfunc_dict = cls.__dict__.get("attr_calcfunc_dict", cls.createAttributeCalculationFunctionDictionary())
        if attr not in attr_calcfunc_dict.keys():
            return False
        calcfunc, b = attr_calcfunc_dict[attr]
        #print(calcfunc, b)
        #calcfunc, b = (calcfunc_tup, False) if isinstance(calcfunc_tup, str) else calcfunc_tup
        res = calcfunc(self)
        if not b:
            self.__dict__[f"_{attr}"] = res
            #print(f"attribute {attr} new value = {res}")
        return True
    
    def setAttributes(self, setattr_dict: Dict[str, Any], _from_container: bool=False, **kwargs) -> None:
        #print("Using ComponentBaseClass method setAttributes()")
        #print(setattr_dict)
        #if "title_color" in setattr_dict.keys():
        #    print(f"setting title_color attribute for {type(self).__name__}")
        #print("using setAttributes()")
        #print(setattr_dict)
        #if "val" in setattr_dict.keys():
        #    print(setattr_dict)
        #print(f"\nsetting attributes {list(setattr_dict.keys())}")
        #print(setattr_dict)
        #print(type(self).__name__)
        #if "_name" in self.__dict__.keys(): print(self.__dict__["_name"])
        #for attr in ["max_shape"]:
        #    if attr in setattr_dict.keys():
        #        print(f"{attr} = {setattr_dict[attr]}")
        cls = type(self)
        attr_calcfunc_dict = cls.getAttributeCalculationFunctionDictionary()
        fixed_attr_set = cls.getFixedAttributeSet()
        #fixed_attr_set = cls.__dict__.get("fixed_attr_set", cls.createFixedAttributeSet())
        
        self.__dict__.setdefault("is_default_set", set())
        is_default_set = self.is_default_set
        for attr, val in setattr_dict.items():
            if val is not None and attr in attr_calcfunc_dict.keys():
                raise AttributeError(f"The attribute '{attr}' of '{cls.__name__}' object cannot be set directly, as its value is calculated from other attribute values")
            elif attr in fixed_attr_set:
                raise AttributeError(f"The attribute '{attr}' of '{cls.__name__}' object cannot be reassigned")
            is_default_set.discard(attr)
        
        attr_deffunc_dict = cls.getAttributeDefaultFunctionDictionary()
        
        if "attr_list" not in cls.__dict__.keys():
            attr_list, attr_dict, reset_graph, custom_reset_funcs = cls.createResetStructures()
        else:
            attr_list = cls.attr_list
            attr_dict = cls.attr_dict
            reset_graph = cls.reset_graph
            custom_reset_funcs = cls.custom_reset_funcs
        inds = []
        
        def setAttrCustom(attr: str, val: Any) -> None:
            #if attr.lstrip("_") in {"val", "val_str", "val_text_surf"}:
            #    print(f"setting attribute {attr} to {val}")
            self.__dict__[attr] = val
            #print(f"self.__dict__['{attr}'] = {val}")
            return
        
        def addBooleanFunction(
            func_lst: Union[List[Callable[["ComponentBaseClass", "ComponentBaseClass"], bool]], bool],
            func: Union[Callable[["ComponentBaseClass", "ComponentBaseClass"], bool], bool]
        ) -> Union[List[Callable[["ComponentBaseClass", "ComponentBaseClass"], bool]], bool]:
            if func_lst is True or func is True:
                return True
            elif func is False:
                return func_lst
            elif func_lst is False:
                return [func]
            func_lst.append(func)
            return func_lst
        
        container_attr_reset_dict = getattr(self, "_container_attr_reset_dict", {})
        #if container_attr_reset_dict:
        #    print(f"container_attr_reset_dict = {container_attr_reset_dict}")
        
        #print(container_attr_reset_dict)
        attr_processing_dict = cls.getAttributeProcessingDictionary()
        container_reset_attrs = {}
        attr_reset_dict = getattr(self, "_attr_reset_funcs", {})
        reset_funcs = []
        for attr, val in setattr_dict.items():
            #print(attr, val)
            
            sub_attr = attr if attr.startswith("_") else f"_{attr}"
            #print(sub_attr)
            #print(sub_attr in self.__dict__.keys(), attr in self.__dict__.keys())
            if attr not in attr_dict.keys():
                pass#setAttrCustom(*attr_tup)
            elif getattr(self, sub_attr, None) != val:
                inds.append(attr_dict[attr])
            elif val is not None:
                continue
            #sub_attr = f"_{attr}"
            val_prev = self.__dict__.get(sub_attr, None)
            #if attr == "max_height0":
            #    print(f"max_height0 val_prev = {val_prev}, val = {val}")
            for func in custom_reset_funcs.get(attr_dict.get(attr, None), []):
                #if attr == "max_height":
                #    print(f"reset function for max_height: {(func, val_prev)}")
                reset_funcs.append((func, val_prev))
            if attr in container_attr_reset_dict.keys():
                for attr2, func in container_attr_reset_dict[attr].items():
                    func_lst = addBooleanFunction(container_reset_attrs.get(attr2, []), func)
                    container_reset_attrs[attr2] = func_lst
            #process_func = attr_processing_dict.get(attr, (lambda x: x))
            setAttrCustom(sub_attr, val)
            #print(attr2 in self.__dict__.keys(), attr in self.__dict__.keys())
        #print("hello")
        def bfs(inds: List[int]) -> Tuple[List[Callable[["DisplayComponentBase"], None]]]:
            reset_funcs = []
            #for idx in inds:
                
            qu = deque(inds)
            seen = set(inds)
            while qu:
                idx = qu.popleft()
                for idx2, funcs in reset_graph[idx].items():
                    if idx2 in seen:
                        continue
                    if isinstance(funcs, bool):
                        b = funcs
                    else:
                        b = True
                        for func in funcs:
                            if not func(self):
                                b = False
                                break
                    if not b: continue
                    attr = attr_list[idx2]
                    #if attr == "updated":
                    #    print("attribute 'updated' reset")
                    attr_ = f"_{attr}"
                    #print(attr, attr_)
                    val_prev = self.__dict__.get(attr_, None)
                    
                    if attr in container_attr_reset_dict.keys():
                        for attr2, func in container_attr_reset_dict[attr].items():
                            func_lst = addBooleanFunction(container_reset_attrs.get(attr2, []), func)
                            container_reset_attrs[attr2] = func_lst
                    setAttrCustom(attr_, None)
                    #if attr in attr_deffunc_dict.keys():
                    #    reset_funcs.extend(custom_reset_funcs.get(idx
                    #    #default_attrs.append(attr)
                    #print(sub_attr)
                    #print(custom_reset_funcs)
                    #print(idx2)
                    for func in custom_reset_funcs.get(idx2, []):
                        reset_funcs.append((func, val_prev))
                    for func in attr_reset_dict.get(attr, []):
                        #if attr == "updated":
                        #    print("reset function for 'updated' added")
                        reset_funcs.append((func, val_prev))
                    #print(attr, funcs)
                    #reset_funcs.extend(funcs)
                    seen.add(idx2)
                    qu.append(idx2)
            return reset_funcs
        reset_funcs.extend(bfs(inds))
        #print("custom functions:")
        #print(funcs)
        #print(f"default_attrs = {default_attrs}")
        #for attr in default_attrs:
        #    print(self.__dict__[f"_{attr}"])
        #    print(getattr(self, attr))
        #print(reset_funcs)
        for func, prev_val in reset_funcs:
            func(self, prev_val)
        #print("hello")
        #print(container_attr_reset_dict, container_reset_attrs)
        if container_reset_attrs and "_container_obj" in self.__dict__.keys():
            container_obj = self._container_obj
            container_setattrs = {}
            for attr, func_lst in container_reset_attrs.items():
                if not func_lst: continue
                if not isinstance(func_lst, bool):
                    for func in func_lst:
                        if func(container_obj, self):
                            break
                    else: continue
                container_setattrs[attr] = None
            #print(f"container_setattrs = {container_setattrs}")
            container_obj.setAttributes(container_setattrs)
        #if "display_surf" in setattr_dict.keys():
        #    print("hola")
        #    print("_display_surf" in self.__dict__.keys(), "display_surf" in self.__dict__.keys())
        return
    
    def __setattr__(self, attr: str, val: Any) -> None:
        if attr and attr[0] == "_":
            self.__dict__[attr] = val
            return
        return self.setAttributes({attr: val})
    
    def __getattr__(self, attr: str) -> Any:
        #if attr == "display_surf":
        #    print(f"Using ComponentBaseClass method __getattr__() for attribute {attr}")
        def notFoundError():
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
        
        def findAttribute(attr: str) -> Tuple[Union[Optional[Any], bool]]:
            if attr in self.__dict__.keys():
                return (self.__dict__[attr], True)
            for cls in type(self).mro():
                if attr in cls.__dict__.keys():
                    return (cls.__dict__[attr], True)
            return (None, False)
        
        #if attr in {"val", "val_str", "val_text_surf"}:
        #    print(f"attempting to get attribute {attr}")
        val, b = findAttribute(attr)
        #if attr == "init_val":
        #    print(attr)
        #    print(attr in self.__dict__.keys())
        #    print(val, b)
        if val is not None: return val
        if attr and attr[0] == "_":
            notFoundError()
        attr2 = f"_{attr}"
        val, b = findAttribute(attr2)
        #if attr == "init_val":
        #    print(attr2)
        #    print(attr2 in self.__dict__.keys())
        #    print(val, b)
        if val is not None: return val
        
        if self.calculateAndSetAttribute(attr):
            return self.__dict__[attr2]
        
        self.__dict__.setdefault("is_default_set", set())
        is_default_set = self.is_default_set
        cls = type(self)
        attr_deffunc_dict = cls.getAttributeDefaultFunctionDictionary()
        if attr in attr_deffunc_dict.keys():
            val = attr_deffunc_dict[attr][0](self)
            self.__dict__[attr2] = val
            is_default_set.add(attr)
            return val
            #print(f"val = {val}")
        #if attr == "init_val":
        #    print("finding init_val")
        #    print(b)
        return val if b else notFoundError()

class ComponentGroupElementBaseClass(ComponentBaseClass):
    finalizer_attributes = {"group_idx", (lambda cls: cls.group_obj_attr)}
    
    def __init__(self, **kwargs):
        #print("hello2")
        #print(kwargs)
        self.__dict__[f"_{self.group_obj_attr}"] = kwargs["_group"]
        super().__init__(**self.initArgsManagement(locals(), kwargs=kwargs, rm_args=["_group"]))
    
    @classmethod
    def remove(cls, inst_idx: int) -> None:
        
        attrs = cls.instances_finalizer_attr_vals[inst_idx]
        #print("\n\n")
        #print(f"attrs = {attrs}")
        #print(f"cls = {cls.__name__}")
        #print(f"mro(cls) = {(cls.mro())}")
        #print(f"cls.instances_finalizer_attr_vals = {cls.instances_finalizer_attr_vals}")
        group = attrs[cls.group_obj_attr]
        group_idx = attrs["group_idx"]
        
        group._removeElementFromRecord(group_idx)
        super().remove(inst_idx)
        return
    
    @classmethod
    def createGroupDeterminedAttributeDict(cls) -> Set[str]:
        group_determined_attr_dict = {}
        # Adding the attributes that are set via the group
        grp_cls = cls.group_cls_func()
        #el_inherit_attr_dict = grp_cls.__dict__.get("el_inherit_attr_dict", grp_cls.createElementInheritedAttributesDictionary())
        el_inherit_attr_dict = grp_cls.getElementInheritedAttributesDictionary()
        for grp_attr, el_attr in el_inherit_attr_dict.items():
            group_determined_attr_dict[el_attr] = grp_attr
        cls.group_determined_attr_dict = group_determined_attr_dict
        return group_determined_attr_dict
    
    def setAttributes(self, setattr_dict: Dict[str, Any], _from_group: bool=False, **kwargs) -> None:
        if not _from_group:
            cls = type(self)
            grp_cls = cls.group_cls_func()
            group_determined_attr_dict = cls.__dict__.get("group_determined_attr_dict", cls.createGroupDeterminedAttributeDict())
            for attr in setattr_dict.keys():
                if attr in group_determined_attr_dict.keys():
                    raise AttributeError(
                        f"The attribute '{attr}' of '{cls.__name__}' "
                        "object can only be reassigned through "
                        "changing the corresponding attribute "
                        f"('{group_determined_attr_dict[attr]}') in "
                        f"the '{grp_cls.__name__}' group object to "
                        "which it belongs. This group object can be "
                        "accessed via the attribute "
                        f"'{cls.group_obj_attr}'"
                    )
        super().setAttributes(setattr_dict, **kwargs)
        return
    
    @classmethod
    def _createFixedAttributeSet(cls) -> Set[str]:
        fixed_attr_set = super()._createFixedAttributeSet()
        fixed_attr_set.add(cls.group_obj_attr)
        return fixed_attr_set

class ComponentGroupBaseClass(ComponentBaseClass):
    
    attribute_default_functions = {
        "elements_weakref": ((lambda obj: []),),
        "available_el_idx_heap": ((lambda obj: []),),
    }
    
    fixed_attributes = {"elements_weakref"}
    
    element_inherited_attributes = {}
    
    def __init__(self, **kwargs):
        #self._elements_weakref = []
        #self._available_el_idx_heap = []
        super().__init__(**self.initArgsManagement(locals(), kwargs=kwargs), _from_group=True)
        
    @classmethod
    def _createElementInheritedAttributesDictionary(cls) -> Dict[str, Any]:
        el_inherit_attr_dict = {}
        for cls2 in cls.mro():
            for grp_attr, el_attr in cls2.__dict__.get("element_inherited_attributes", {}).items():
                el_inherit_attr_dict.setdefault(grp_attr, el_attr)
        #cls.el_inherit_attr_dict = el_inherit_attr_dict
        return el_inherit_attr_dict
    
    @classmethod
    def getElementInheritedAttributesDictionary(cls) -> Dict[str, Any]:
        if cls.__dict__.get("_el_inherit_attr_dict", None) is None:
            cls._el_inherit_attr_dict = cls._createElementInheritedAttributesDictionary()
            #print(f"set el_inherit_attr_dict for class {cls.__name__}")
        return cls._el_inherit_attr_dict
    
    @classmethod
    def createResetStructures(cls) -> Tuple[Union[List[str], Dict[str, int], List[Dict[int, Union[List[Callable[["DisplayComponentBase"], bool]]]]], Dict[int, List[Callable[[], None]]]]]:
        (attr_list, attr_dict, reset_graph, custom_reset_funcs) = super().createResetStructures()
        #el_inherit_attr_dict = cls.__dict__.get("el_inherit_attr_dict", cls.createElementInheritedAttributesDictionary())
        el_inherit_attr_dict = cls.getElementInheritedAttributesDictionary()
        
        def resetFunction(grp_attr: str, el_attr: str, obj: "ComponentGroupBaseClass", prev_val: Any) -> None:
            #print(f"calling resetFunction with group attribute {grp_attr} and element attribute {el_attr} for a {type(obj).__name__} object")
            val = getattr(obj, grp_attr)
            #print(f"val = {val}")
            for el_weakref in obj.elements_weakref:
                if el_weakref is None:
                    continue
                el = el_weakref()
                #print(el_weakref)
                #print(el)
                el.setAttributes({el_attr: val}, _from_group=True)
            return
        
        for grp_attr, el_attr in el_inherit_attr_dict.items():
            idx = cls._attr2Index(grp_attr, attr_list, attr_dict, reset_graph)
            custom_reset_funcs.setdefault(idx, [])
            custom_reset_funcs[idx].append(functools.partial(resetFunction, grp_attr, el_attr))
        
        return (attr_list, attr_dict, reset_graph, custom_reset_funcs)
    
    def _setMemberInheritedAttributes(self, set_group_attrs: Optional[Set[str]]=None) -> None:
        el_inherit_attr_dict = self.getElementInheritedAttributesDictionary()#cls.__dict__.get("el_inherit_attr_dict", cls.createElementInheritedAttributesDictionary())
        el_setattr_dict = {}
        if set_group_attrs is None:
            set_group_attrs = el_inherit_attr_dict.keys()
        for grp_attr in set_group_attrs:
            if grp_attr not in el_inherit_attr_dict.keys() or not hasattr(self, grp_attr): continue
            el_attr = el_inherit_attr_dict[grp_attr]
            el_setattr_dict[el_attr] = getattr(self, grp_attr)
        #for grp_attr, val in setattr_dict.items():
        #    if grp_attr in el_inherit_attr_dict.keys():
        #        el_setattr_dict[el_inherit_attr_dict[grp_attr]] = val
        for el_weakref in self.elements_weakref:
            if el_weakref is None:
                continue
            el = el_weakref()
            el.setAttributes(el_setattr_dict, _from_group=True)
        return
    
    def setAttributes(self, setattr_dict: Dict[str, Any], **kwargs) -> None:
        super().setAttributes(setattr_dict)
        self._setMemberInheritedAttributes(setattr_dict.keys())
        return
        """
        cls = type(self)
        el_inherit_attr_dict = self.getElementInheritedAttributesDictionary()#cls.__dict__.get("el_inherit_attr_dict", cls.createElementInheritedAttributesDictionary())
        el_setattr_dict = {}
        for grp_attr, val in setattr_dict.items():
            if grp_attr in el_inherit_attr_dict.keys():
                el_setattr_dict[el_inherit_attr_dict[grp_attr]] = val
        for el_weakref in self.elements_weakref:
            if el_weakref is None:
                continue
            el = el_weakref()
            el.setAttributes(el_setattr_dict, _from_group=True)
        return
        """
    

    
    def __len__(self) -> int:
        return len(self.elements_weakref) - len(self.available_el_idx_heap)
    
    def _addElement(self,
        _from_group: bool=True,
        **kwargs,
    ) -> "ComponentGroupElementBaseClass":
        group_element_cls = type(self).group_element_cls_func()
        grp_attr = group_element_cls.group_obj_attr
        #kwargs2 = {}
        #for 
        res = group_element_cls(**kwargs, **{grp_attr: self, "_from_group": True})
        self._addElementToRecord(res)
        self._setMemberInheritedAttributes(set_group_attrs=None)
        return res
    
    def _addElementToRecord(self, element: "ComponentGroupElementBaseClass") -> int:
        """
        idx = None
        while self.available_el_idx_heap:
            idx = heapq.heappop(self.available_el_idx_heap)
            if idx >= len(self.elements_weakref):
                self.available_el_idx_heap = []
                break
            if self.elements_weakref[idx] is not None:
                break
        else:
            idx = len(self.elements_weakref)
            self.elements_weakref.append(None)
        """
        idx = findSmallestAvailableIndex(
            self.elements_weakref,
            self.available_el_idx_heap,
        )
        #print(f"\nSetting group_idx to {idx}")
        element.group_idx = idx
        #print(f"group_idx = {element.group_idx}")
        inst_idx = element.inst_idx
        #print(f"finalizer_attr_vals = {element.instances_finalizer_attr_vals[inst_idx]}")
        self.elements_weakref[idx] = weakref.ref(element)
        
        return idx
    
    def _removeElementFromRecord(self, group_idx: int) -> None:
        popIndex(group_idx, self.elements_weakref, self.available_el_idx_heap)
        return
    
        """
        #idx = element._group_idx
        if group_idx < len(self.elements_weakref):
            self.elements_weakref[group_idx] = None
            heapq.heappush(self.available_el_idx_heap, group_idx)
            return
        self.elements_weakref.pop()
        while self.elements_weakref and self.elements_weakref[-1] is None:
            self.elements_weakref.pop()
        return
        """
        

class DisplayComponentBase(ComponentBaseClass):
    
    reset_graph_edges = {
        "shape": {"topleft": (lambda obj: obj.anchor_type != "topleft")},
        "anchor_pos": {"topleft": True},
        "anchor_type": {"topleft": True},
        "topleft": {"topleft_screen": True},
        "screen_topleft_offset": {"topleft_screen": True},
    }
    custom_reset_methods = {}
    
    attribute_calculation_methods = {
        "topleft": "calculateTopLeft",#"calculateAndSetTopLeft",
        "topleft_screen": "calculateTopLeftScreen",#"calculateAndSetTopLeftScreen",
    }
    
    attribute_default_functions = {
        "anchor_type": ((lambda obj: "topleft"),),
        "screen_topleft_offset": ((lambda obj: (0, 0)),),
    }
    
    fixed_attributes = set()
    
    def __init__(self, shape: Tuple[Real],\
            anchor_pos: Tuple[Real], anchor_type: Optional[str]=None,\
            screen_topleft_offset: Optional[Tuple[Real]]=None, **kwargs):
        
        super().__init__(**self.initArgsManagement(locals(), kwargs=kwargs))
        pg.init()
    
    
    def findScreenPosition(self, surf_pos: Tuple[Real]):
        return tuple(x + y for x, y in zip(surf_pos, self.screen_topleft_offset))
    
    def calculateTopLeft(self) -> Tuple[int]:
        #print(type(self), self._shape)
        return topLeftFromAnchorPosition(self.shape, self.anchor_type,\
                self.anchor_pos)
    
    def calculateTopLeftScreen(self):
        return self.findScreenPosition(self.topleft)
    
    @abstractmethod
    def draw(self) -> None:
        pass
    

def keysDownFunction0(obj: "InteractiveDisplayComponentBase") -> Set[int]:
    #print("Using keysDownFunction0()")
    res = set()
    if obj.enter_keys_enablement[0]:
        res |= obj.enter_keys
    if obj.navkeys_enablement[0]:
        res |= obj.navkeys_dict.keys()
    return res

def keyEventFilter0(obj: "InteractiveDisplayComponentBase", event, navkeys_enabled: bool=True, enter_keys_enabled: bool=True) -> bool:
    #print("Using keyEventFilter0()")
    if navkeys_enabled and event.key in obj.navkeys_dict.keys():
        return True
    if enter_keys_enabled and event.key in obj.enter_keys:
        return True
    return False

def mouseEventFilter0(obj: "InteractiveDisplayComponentBase", event, mouse_enabled: bool=True) -> bool:
    #print(f"mouse_enabled = {mouse_enabled}")
    #print(f"event.button = {event.button}")
    return mouse_enabled and event.button == 1

class InteractiveDisplayComponentBase(DisplayComponentBase):
    
    #change_reset_attrs = {"anchor": ["ranges_surf", "ranges_screen"], "shape": ["ranges_surf", "ranges_screen"], "screen_topleft_offset": ["ranges_screen"]}
    
    reset_graph_edges = {
        "shape": {"ranges_surf": True},
        "anchor_pos": {"ranges_surf": True},
        "anchor_type": {"ranges_surf": True},
        "ranges_surf": {"ranges_screen": True},
        "screen_topleft_offset": {"ranges_screen": True},
        "navkeys_dict": {"navkeys": True},
    }
    
    custom_reset_methods = {}
    
    attribute_calculation_methods = {
        "ranges_surf": "calculateRangesSurface",
        "ranges_screen": "calculateRangesScreen",
    }
    
    attribute_default_functions = {
        "navkeys": ((lambda obj: navkeys_def_glob),),
        "enter_keys": ((lambda obj: enter_keys_def_glob),),
        "mouse_enablement": ((lambda obj: (False, False, False)),),
        "navkeys_enablement": ((lambda obj: (False, False, False)),),
        "enter_keys_enablement": ((lambda obj: (False, False, False)),),
    }
    
    fixed_attributes = set()
    
    _user_input_processor = UserInputProcessor(keys_down_func=keysDownFunction0,
            key_press_event_filter=lambda obj, event: keyEventFilter0(obj, event, navkeys_enabled=obj.navkeys_enablement[1], enter_keys_enabled=obj.enter_keys_enablement[1]),
            key_release_event_filter=lambda obj, event: keyEventFilter0(obj, event, navkeys_enabled=obj.navkeys_enablement[2], enter_keys_enabled=obj.enter_keys_enablement[2]),
            mouse_press_event_filter=lambda obj, event: mouseEventFilter0(obj, event, mouse_enabled=obj.mouse_enablement[1]),
            mouse_release_event_filter=lambda obj, event: mouseEventFilter0(obj, event, mouse_enabled=obj.mouse_enablement[2]),
            other_event_filter=False,
            get_mouse_status_func=(lambda obj: obj.mouse_enablement[0]))
    
    
    def __init__(self, shape: Tuple[Real],\
            anchor_pos: Tuple[Real], anchor_type: str="topleft",\
            screen_topleft_offset: Tuple[Real]=(0, 0),\
            mouse_enablement: Optional[Tuple[bool]]=None,\
            navkeys_enablement: Optional[Tuple[bool]]=None,\
            navkeys: Optional[Tuple[Tuple[Set[int]]]]=None,\
            enter_keys_enablement: Optional[Tuple[bool]]=None,\
            enter_keys: Optional[Set[int]]=None,\
            keys_down_func=False,\
            key_press_event_filter=False,\
            key_release_event_filter=False,\
            mouse_press_event_filter=False,\
            mouse_release_event_filter=False,\
            other_event_filter=False,\
            get_mouse_status_func=False,\
            **kwargs):
        
        super().__init__(**self.initArgsManagement(locals(), kwargs=kwargs))
        type(self)._resolveClassUserInputProcessors()
    
    @classmethod
    def _resolveClassUserInputProcessors(cls) -> None:
        #print(cls.__name__)
        if cls.__dict__.get("_user_input_processor_resolved", False):
            return
        if "_user_input_processor" not in cls.__dict__.keys():
            cls._user_input_processor = UserInputProcessor(
                keys_down_func=False,
                key_press_event_filter=False,
                key_release_event_filter=False,
                mouse_press_event_filter=False,
                mouse_release_event_filter=False,
                other_event_filter=False,
                get_mouse_status_func=False,
            )
        
        for cls2 in cls.__mro__[1:-2]:
            sub_uip = cls2.__dict__.get("_user_input_processor", None)
            if sub_uip is None: continue
            cls._user_input_processor.addSubUIP(sub_uip)
        
        sub_components_dict = cls.__dict__.get("sub_components_dict", cls.createSubComponentDictionary())
        for component_attr in sub_components_dict.keys():
            #print("hello")
            #print(component_attr)
            cls2 = sub_components_dict[component_attr]["class"]
            if not hasattr(cls2, "_resolveClassUserInputProcessors"):
                continue
            #print("hi")
            cls2._resolveClassUserInputProcessors()
            sub_uip = cls2.__dict__.get("_user_input_processor", None)
            if sub_uip is None: continue
            #print("hi2")
            #print(cls2)
            cls._user_input_processor.addSubUIP(sub_uip, obj_func=lambda obj: getattr(obj, component_attr))
        cls._user_input_processor_resolved = True
        #print("howdy")
        #print(cls._user_input_processor.mouse_press_event_filter)
        #print(cls._user_input_processor.key_press_event_filter)
        return
    
    def calculateRangesSurface(self) -> Tuple[Tuple[Real]]:
        return tuple((0, x) for x in self.shape)
    
    def findNavkeyDict(self) -> Dict[int, Tuple[int]]:
        return createNavkeyDict(self.navkeys)
    
    def rangesSurface2RangesScreen(self, ranges: Tuple[Tuple[Real]]) -> Tuple[Tuple[Real]]:
        #print("hi")
        offset = self.topleft_screen
        if offset == (0, 0): return ranges
        return tuple(tuple(x + y for x in rng) for y, rng in zip(offset, ranges))
    
    def calculateRangesScreen(self) -> Tuple[Tuple[Real]]:
        return self.rangesSurface2RangesScreen(self.ranges_surf)
    
    def mouseOverSurface(self, mouse_pos: Optional[Tuple]=None, check_axes: Tuple[int]=(0, 1)) -> bool:
        if not self.mouse_enablement[0] or not pg.mouse.get_focused():
            return False
        if mouse_pos is None:
            mouse_pos = pg.mouse.get_pos()
        
        ranges = self.ranges_screen
        for i in check_axes:
            x, rng = mouse_pos[i], ranges[i]
            if x < rng[0] or x > rng[1]:
                return False
        return True
    
    def getRequiredInputs(self) -> Tuple[Union[bool, Dict[str, Union[List[int], Tuple[Union[Tuple[int], int]]]]]]:
        #print("Using InteractiveDisplayComponent method getRequiredInputs()")
        #print(self.user_input_processor.get_mouse_status_func_actual(self), self.user_input_processor.get_mouse_status_func(self), self, getattr(self._user_input_processor, "get_mouse_status_func0", None))
        quit, esc_pressed, events = self.user_input_processor.getEvents(self)
        return quit, esc_pressed, {"events": events,\
                "keys_down": self.user_input_processor.getKeysHeldDown(self),\
                "mouse_status": self.user_input_processor.getMouseStatus(self)}
    
    @classmethod
    def createInteractiveSubComponentsSet(cls) -> Set[str]:
        sub_components_dict = cls.__dict__.get("sub_components_dict", cls.createSubComponentDictionary())
        interactive_sub_components_set = set()
        for attr, sc_dict in sub_components_dict.items():
            cls2 = sc_dict["class"]
            if issubclass(cls2, InteractiveDisplayComponentBase):
                interactive_sub_components_set.add(attr)
        cls.interactive_sub_components_set = interactive_sub_components_set
        return interactive_sub_components_set
    
    def eventLoopComponents(
        self,
        events: List[int],
        keys_down: List[int],
        mouse_status: Tuple[int],
        check_axes: Tuple[int],
    ) -> Tuple[bool, bool, bool, Any]:
        cls = type(self)
        interactive_sub_components_set = cls.__dict__.get("interactive_sub_components_set", cls.createInteractiveSubComponentsSet())
        quit, running, screen_changed = False, True, False
        val_dict = {}
        for attr in interactive_sub_components_set:
            #print(attr)
            component = getattr(self, attr)
            #print(component)
            quit2, running2, screen_changed2, val2 = component.eventLoop(events=events, keys_down=keys_down, mouse_status=mouse_status, check_axes=check_axes)
            quit = quit or quit2
            running = running and running2
            screen_changed = screen_changed or screen_changed2
            val_dict[attr] = val2
        return quit, running, screen_changed, val_dict
    
    def getEventLoopArguments(
        self,
        events: Optional[List[int]]=None,
        keys_down: Optional[List[int]]=None,
        mouse_status: Optional[Tuple[int]]=None,
        check_axes: Tuple[int]=(0, 1),
    ) -> Tuple[Tuple[bool, bool], Tuple[List[int], List[int], Tuple[int], Tuple[int]]]:
        
        quit = False
        esc_pressed = False
        
        uip = self._user_input_processor
        
        if events is None:
            quit, esc_pressed, events = uip.getEvents()
        
        if keys_down is None:
            keys_down = uip.getKeysHeldDown()
        
        if mouse_status is None:
            mouse_status = uip.getMouseStatus()
        
        return ((quit, esc_pressed), (events, keys_down, mouse_status, check_axes))
    
    @abstractmethod
    def eventLoop(
        self,
        events: Optional[List[int]]=None,
        keys_down: Optional[List[int]]=None,
        mouse_status: Optional[Tuple[int]]=None,
        check_axes: Tuple[int]=(0, 1),
    ) -> Tuple[bool, bool, bool, Any]:
        pass
    """    
    def eventLoop(
        self,
        events: Optional[List[int]]=None,
        keys_down: Optional[List[int]]=None,
        mouse_status: Optional[Tuple[int]]=None,
        check_axes: Tuple[int]=(0, 1),
    ) -> Tuple[bool, bool, bool, Any]:
        
        ((quit, esc_pressed), (events, keys_down, mouse_status, check_axes)) = self.getEventLoopArguments(events=events, keys_down=keys_down, mouse_status=mouse_status, check_axes=check_axes)
        running = not quit and not esc_pressed
    
        quit2, running2, screen_changed, val_dict = self.eventLoopComponents(
            events=events,
            keys_down=keys_down,
            mouse_status=mouse_status,
            check_axes=check_axes,
        )
        quit = quit or quit2
        running = running and running2
        
        return (quit, running, screen_changed, val_dict)
    """
