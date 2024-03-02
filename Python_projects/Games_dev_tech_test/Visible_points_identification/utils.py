#!/usr/bin/env python3

import random
import unittest

from abc import ABC, abstractmethod
from typing import (
    Union,
    Optional,
    Generator,
    Tuple,
)

from .visible_point_identification import PointSet

Real = Union[int, float]

def randomCoordinateGenerator(
    coord_ranges: Tuple[Tuple[Real]],
    n_points: Optional[int]=None,
) -> Generator[Tuple[float], None, None]:
    
    """
    Generator that yields coordinates in n-dimensional space (where
    n is the length of the argument coord_ranges), such that the
    values of each component is a float drawn from a continuous random
    uniform distribution in the corresponding range in coord_ranges
    
    Args:
        Required positional:
        coord_ranges (tuple of 2-tuples of ints/floats): The ranges
                of values each component of the coordinates can take,
                in the same order as the order of the components.
                The range is closed at the lower end and open at the
                upper end
        
        Optional named:
        n_points (int or None): The number of points to be generated.
                If given as None, generates coordinates endlessly
                (and so the generator must be terminated in the loop
                used with a break or return statement)
            Default: None
    
    Yields:
    n-tuples (where n is the length of coord_ranges) of float values,
    such that each component is inside the range given by coord_ranges.
    """
    
    def endlessIterator() -> Generator[None, None, None]:
        while True: yield None
        return

    iter_obj = range(n_points) if isinstance(n_points, int)\
            else endlessIterator()
    
    offsets = [rng[0] for rng in coord_ranges]
    mults = [rng[1] - rng[0] for rng in coord_ranges]
    
    for _ in iter_obj:
        yield tuple(random.random() * mult + offset\
                for mult, offset in zip(mults, offsets))
    return

def randomPointSetCreator(
    x_range: Tuple[Real],
    y_range: Tuple[Real],
    n_points: int,
) -> "PointSet":
    """
    Creates PointSet instance for a given number of random points
    in 2-dimensional Euclidean space, each with a random named
    associated direction ('North', 'South', 'East' or 'West',
    each randomly selected with equal probability)
    
    Random values are sampled from a uniform distribution in the
    relevant range, where the lower end of the range is closed
    and the upper end is open
    
    Args:
        x_range (2-tuple of ints/floats): The range of x-values
                allowed for the positions of the points when
                expressed in Cartesian coordinates
        y_range (2-tuple of ints/floats): The range of y-values
                allowed for the positions of the points when
                expressed in Cartesian coordinates
        n_points (int): The number of random points to be generated
    
    Returns:
    A PointSet object with n_points points, where the positions
    of each point in Cartesian coordinates has its x-coordinate in
    the range x_range and its y-coordinate in the range y_range
    """
    directs = list(PointSet.direct_dict.keys())
    
    n_direct = len(directs)
    points = []
    for i, coords in enumerate(randomCoordinateGenerator(\
            (x_range, y_range), n_points=n_points), start=5):
        direct_idx = random.randrange(0, n_direct)
        direct = directs[direct_idx]
        points.append((coords, i, direct))
    return PointSet(points)
