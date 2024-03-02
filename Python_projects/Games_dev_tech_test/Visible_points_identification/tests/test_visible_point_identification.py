#!/usr/bin/env python3
import math
import os
import random
import sys
import unittest

from typing import (
    Callable,
    Optional,
    Union,
    Dict,
    Tuple,
    List,
    Set,
    Any,
    Iterable,
)

sys.path.append(os.path.abspath('../../'))

from Visible_points_identification import (
    PointSet,
)

module_name = "Visible_points_identification"

Real = Union[int, float]

def toString(x, not_str: bool=False) -> str:
    qstr = '"' if not not_str and isinstance(x, str) else ''
    return f"{qstr}{x}{qstr}"

def argsKwargsStrings(args: Optional[Tuple[Any]], kwargs: Optional[Dict[str, Any]]=None) -> List[str]:
    res = []
    if args:
        res.append(", ".join([toString(x) for x in args]))
    if kwargs:
        res.append(", ".join([f"{toString(x, not_str=True)}={toString(y)}" for x, y in kwargs.items()]))
    return res

class TestPointSet(unittest.TestCase):
    cls = PointSet
    cls_name = cls.__name__
    cls_name_full = f"{module_name}.{cls_name}"
    
    eps = 10 ** -5
    
    n_random_sample = 100
    
    def angleVector_chk(
        self,
        method: Callable,
        args: Tuple[Any],
        kwargs: Dict[str, Any],
    ) -> Tuple[bool, Any]:
        angle = args[0] if args else kwargs["angle"]
        vec = method(*args, **kwargs)
        
        if not hasattr(vec, "__len__"): return False, vec
        if len(vec) != 2: return False, vec
        try:
            length = sum(x ** 2 for x in vec)
        except:
            return False, vec
        if abs(length - 1) > self.eps: return False, vec
        angle2 = math.degrees(math.atan2(vec[1], vec[0]))
        diff = (angle - angle2) % 360
        if diff > 180: diff -= 360
        if abs(diff) > self.eps: return False, vec
        return True, vec
    
    def template_test_angleVector(self, args_kwargs_iter: Iterable) -> None:
        cls_method = self.cls.angleVector
        cls_method_name = cls_method.__name__
        cls_method_name_full = f"{self.cls_name}.{cls_method_name}"
        chk_func = self.angleVector_chk
        for args, kwargs in args_kwargs_iter:
            args_kwargs_str = ", ".join(argsKwargsStrings(args, kwargs))
            #print(args_kwargs_str)
            chk, calc = chk_func(cls_method, args, kwargs)
            with self.subTest(msg=f"Call of class method "\
                    f"{cls_method_name_full}({args_kwargs_str})."):
                self.assertTrue(
                    chk,
                    msg=f"{cls_method_name_full}({args_kwargs_str}) "
                            f"gave the value {calc}, which is not an "
                            "acceptable answer."
                )
        return
    
    def test_angleVector_specific(self) -> None:
        args_kwargs_iter = list(((x,), {}) for x in range(-15 * 100, 15 * 101, 15))
        return self.template_test_angleVector(args_kwargs_iter)
    
    def test_angleVector_random(self) -> None:
        mn, mx = -10000, 10000
        args_kwargs_iter = ((((mx - mn) * random.random() + mn,), {}) for _ in range(self.n_random_sample))
        return self.template_test_angleVector(args_kwargs_iter)
    
    def vectorProduct_chk(
        self,
        method: Callable,
        args: Tuple[Any],
        kwargs: Dict[str, Any],
    ) -> Tuple[bool, Real, Real, Tuple[Tuple[Real, Real], Tuple[Real, Real]]]:
        # Checks the dot/cross product value with that calculated
        # using polar coordinates (i.e. vector length and angle). Note
        # that here, angles are in radians rather than degrees
        # The checks for dot and cross products are combined
        # as the only difference in the calculation with polar
        # coordinates is the trigonometric function used to
        # process the relative angle (cosine vs sine respectively).
        len1, angle1, len2, angle2 = args
        vec1 = (len1 * math.cos(angle1), len1 * math.sin(angle1))
        vec2 = (len2 * math.cos(angle2), len2 * math.sin(angle2))
        calc = method(vec1, vec2)
        trig_func = math.cos if method.__name__ == "dotProduct" else math.sin
        exp = len1 * len2 * trig_func(angle2 - angle1)
        return abs(calc - exp) <= self.eps, calc, exp, (vec1, vec2)
    
    def template_test_vectorProduct(self, cls_method: Callable, args_kwargs_iter: Iterable) -> None:
        cls_method_name = cls_method.__name__#"dotProduct"
        #print(cls_method_name)
        cls_method_name_full = f"{self.cls_name}.{cls_method_name}"
        chk_func = self.vectorProduct_chk
        for args, kwargs in args_kwargs_iter:
            chk, calc, exp, args2 = chk_func(cls_method, args, kwargs)
            args_kwargs_str = ", ".join(argsKwargsStrings(args2))
            with self.subTest(msg=f"Call of class method "\
                    f"{cls_method_name_full}({args_kwargs_str})."):
                self.assertTrue(
                    chk,
                    msg=f"{cls_method_name_full}({args_kwargs_str}) "
                            f"gave the value {calc}, which is not "
                            "sufficiently close to the expected value "
                            f"{exp}."
                )
        return
    
    def template_test_vectorProduct_random(self, test_method: Callable) -> None:
        mn_len, mx_len = 0, 500
        mn_angle, mx_angle = -100, 100
        
        len_rng = mx_len - mn_len
        angle_rng = mx_angle - mn_angle
        random_len_func = (lambda: random.random() * len_rng + mn_len)
        random_angle_func = (lambda: random.random() * angle_rng + mn_angle)
        
        args_kwargs_iter = (((random_len_func(), random_angle_func(), random_len_func(), random_angle_func()), {}) for _ in range(self.n_random_sample))
        return test_method(args_kwargs_iter)
    
    def template_test_dotProduct(self, args_kwargs_iter: Iterable) -> None:
        return self.template_test_vectorProduct(self.cls.dotProduct, args_kwargs_iter)
    
    def test_dotProduct_specific(self) -> None:
        args_kwargs_iter = [
            ((1, 1, 1, 1), {}),
            ((1, 1, 1, -1), {}),
            ((1, -1, 1, 1), {}),
            ((1, math.pi / 2, 1, 0), {}),
            ((1, 0, 1, math.pi / 2), {}),
            ((1, 0, 1, math.pi), {}),
            ((1, 0, 1, math.pi), {}),
            ((5, -1, 2, 1), {}),
            ((0, 1, 1, -1), {}),
        ]
        return self.template_test_dotProduct(args_kwargs_iter)
    
    def test_dotProduct_random(self) -> None:
        return self.template_test_vectorProduct_random(self.template_test_dotProduct)
    
    def template_test_crossProduct(self, args_kwargs_iter: Iterable) -> None:
        return self.template_test_vectorProduct(self.cls.crossProduct, args_kwargs_iter)
    
    def test_crossProduct_specific(self) -> None:
        args_kwargs_iter = [
            ((1, 1, 1, 1), {}),
            ((1, 1, 1, -1), {}),
            ((1, -1, 1, 1), {}),
            ((1, math.pi / 2, 1, 0), {}),
            ((1, 0, 1, math.pi / 2), {}),
            ((1, 0, 1, math.pi), {}),
            ((1, 0, 1, math.pi), {}),
            ((5, -1, 2, 1), {}),
            ((0, 1, 1, -1), {}),
        ]
        return self.template_test_crossProduct(args_kwargs_iter)
    
    def test_crossProduct_random(self) -> None:
        return self.template_test_vectorProduct_random(self.template_test_crossProduct)
    
    def lengthSquared_chk(
        self,
        method: Callable,
        args: Tuple[Any],
        kwargs: Dict[str, Any],
    ) -> Tuple[bool, Real, Real, Tuple[Real, Real]]:
        # Checks the length squared value given by the class method
        # with that calculated using polar coordinates (i.e. vector
        # length and angle). Note that here, angles are in radians
        # rather than degrees
        length, angle = args
        vec = (length * math.cos(angle), length * math.sin(angle))
        calc = method(vec)
        exp = length ** 2
        return abs(calc - exp) <= self.eps, calc, exp, vec
    
    def template_test_lengthSquared(
        self,
        args_kwargs_iter: Iterable
    ) -> None:
        cls_method = self.cls.lengthSquared
        cls_method_name = cls_method.__name__
        cls_method_name_full = f"{self.cls_name}.{cls_method_name}"
        chk_func = self.lengthSquared_chk
        for args, kwargs in args_kwargs_iter:
            chk, calc, exp, args2 = chk_func(cls_method, args, kwargs)
            args_kwargs_str = ", ".join(argsKwargsStrings(args2))
            with self.subTest(msg=f"Call of class method "\
                    f"{cls_method_name_full}({args_kwargs_str})."):
                self.assertTrue(
                    chk,
                    msg=f"{cls_method_name_full}({args_kwargs_str}) "
                            f"gave the value {calc}, which is not "
                            "sufficiently close to the expected value "
                            f"{exp}."
                )
        return
    
    def test_lengthSquared_specific(self) -> None:
        args_kwargs_iter = [
            ((1, 0), {}),
            ((1, 1), {}),
            ((1, -1), {}),
            ((1, math.pi / 2), {}),
            ((1, math.pi / 3), {}),
            ((5, math.pi * 50), {}),
        ]
        return self.template_test_lengthSquared(args_kwargs_iter)
    
    def test_lengthSquared_random(self) -> None:
        mn_len, mx_len = 0, 500
        mn_angle, mx_angle = -100, 100
        
        len_rng = mx_len - mn_len
        angle_rng = mx_angle - mn_angle
        random_len_func = (lambda: random.random() * len_rng + mn_len)
        random_angle_func = (lambda: random.random() * angle_rng + mn_angle)
        
        args_kwargs_iter = (((random_len_func(), random_angle_func()), {}) for _ in range(self.n_random_sample))
        return self.template_test_lengthSquared(args_kwargs_iter)
    
    def pointInVectorWedgeCalculator_chk(
        self,
        method: Callable,
        args: Tuple[Any],
        kwargs: Dict[str, Any],
    ) -> Tuple[bool, bool, bool, ]:
        # Checks the _inVectorWedge() class method by considering
        # the wedge vectors and the position vector in terms of polar
        # coordinates (i.e. vector length and angle). Note
        # that here, angles are in radians rather than degrees
        cls = self.cls
        len1, angle1, len2, angle2, len3, angle3 = args
        
        vec1, vec2, pos = [(l * math.cos(ang), l * math.sin(ang))\
                for (l, ang) in\
                ((len1, angle1), (len2, angle2), (len3, angle3))]
        cp1, cp2 = [cls.crossProduct(vec, pos) for vec in (vec1, vec2)]
        
        calc = method(vec1, vec2, pos, cp1, cp2)
        
        if len3:
            rot = 2 * math.pi
            angle1_ = angle1 + rot
            while angle2 < angle1:
                angle2 += rot
            while angle2 >= angle1_:
                angle2 -= rot
            while angle3 < angle1:
                angle3 += rot
            while angle3 >= angle1_:
                angle3 -= rot
            exp = (angle3 <= angle2)
        else:
            exp = True
        return (calc == exp), calc, exp, (vec1, vec2, pos, cp1, cp2)
    
    def template_test_pointInVectorWedgeCalculator(
        self,
        args_kwargs_iter: Iterable
    ) -> None:
        cls_method = self.cls._pointInVectorWedgeCalculator
        cls_method_name = cls_method.__name__
        cls_method_name_full = f"{self.cls_name}.{cls_method_name}"
        chk_func = self.pointInVectorWedgeCalculator_chk
        for args, kwargs in args_kwargs_iter:
            chk, calc, exp, args2 = chk_func(cls_method, args, kwargs)
            args_kwargs_str = ", ".join(argsKwargsStrings(args2))
            with self.subTest(msg=f"Call of class method "\
                    f"{cls_method_name_full}({args_kwargs_str})."):
                self.assertTrue(
                    chk,
                    msg=f"{cls_method_name_full}({args_kwargs_str}) "
                            f"gave the value {calc}, when it should "
                            f"have been {exp}."
                )
        return
    
    def test_pointInVectorWedgeCalculator_specific(self) -> None:
        args_kwargs_iter = [
            ((1, 0, 1, math.pi / 2, 1, math.pi / 4), {}),
            ((1, 0, 1, math.pi / 2, 1, 0), {}),
            ((1, 0, 1, math.pi / 2, 1, math.pi / 2), {}),
            ((1, 0, 1, math.pi / 2, 1, math.pi), {}),
            ((1, 0, 1, math.pi / 2, 0, math.pi), {}),
            ((1, 1, 1, 1, 1, 1), {}),
            ((1, 1, 1, 1, 1, 2), {}),
            ((1, 1, 1, 1, 1, 1 + math.pi), {}),
            ((1, 1, 1, 1, 0, 1 + math.pi), {}),
            ((1, 1, 1, 1, 0, 1 + math.pi / 2), {}),
        ]
        return self.template_test_pointInVectorWedgeCalculator(args_kwargs_iter)
    
    def test_pointInVectorWedgeCalculator_random(self) -> None:
        mn_len, mx_len = 0, 500
        mn_angle, mx_angle = -100, 100
        
        len_rng = mx_len - mn_len
        angle_rng = mx_angle - mn_angle
        random_len_func = (lambda: random.random() * len_rng + mn_len)
        random_angle_func = (lambda: random.random() * angle_rng + mn_angle)
        
        args_kwargs_iter = (((random_len_func(), random_angle_func(), random_len_func(), random_angle_func(), random_len_func(), random_angle_func()), {}) for _ in range(self.n_random_sample))
        return self.template_test_pointInVectorWedgeCalculator(args_kwargs_iter)
    
    # TODO- Tests of:
    #    calculateNormalisedBoundingBox,
    #    _filterPoints,
    #    visiblePoints,
    #    otherVisiblePoints,
    #    isVisibleArbitraryPointsAndQueries
