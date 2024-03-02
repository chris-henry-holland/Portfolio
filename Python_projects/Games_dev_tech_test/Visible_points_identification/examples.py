#!/usr/bin/env python3
import math

from sortedcontainers import SortedList

from typing import (
    Optional,
    Union,
    Tuple,
    List,
    Set,
    Dict,
    Any,
    Hashable,
)

from .visible_point_identification import PointSet

Real = Union[int, float]

preset_pointsets = {}
def getPointSet1(eps: Optional[float]=None) -> "PointSet":
    """
    Creates a PointSet object representing the points and
    associated directions given in the original problem
    description (see file Original_problem_description.pdf in
    the Games_dev_tech_test directory)
    
    Args:
        Optional named:
        eps (float): Small strictly positive value used by the PointSet
                object to determine which points are borderline
                visible in a given visibility query (if this is
                not given specifically in the method call)
            Default: class attribute eps (initially 10 ** -5)
    
    Returns:
    PointSet object representing the points and associated directions
    given in the original problem description
    """
    point_set1 = preset_pointsets.get("point_set1", None)
    if point_set1 is not None:
        point_set1.eps = eps
        return point_set1
    points = [
        ((28, 42), 1, "North"),
        ((27, 46), 2, "East"),
        ((16, 22), 3, "South"),
        ((40, 50), 4, "West"),
        ((8, 6), 5, "North"),
        ((6, 19), 6, "East"),
        ((28, 5), 7, "South"),
        ((39, 36), 8, "West"),
        ((12, 34), 9, "North"),
        ((36, 20), 10, "East"),
        ((22, 47), 11, "South"),
        ((33, 19), 12, "West"),
        ((41, 18), 13, "North"),
        ((41, 34), 14, "East"),
        ((14, 29), 15, "South"),
        ((6, 49), 16, "West"),
        ((46, 50), 17, "North"),
        ((17, 40), 18, "East"),
        ((28, 26), 19, "South"),
        ((2, 12), 20, "West"),
    ]
    point_set1 = PointSet(points, eps=eps)
    preset_pointsets["point_set1"] = point_set1
    return point_set1

def isVisible(
    name: Hashable,
    wedge_angle: Real,
    max_dist: Real,
) -> Set[Hashable]:
    """
    This is the function requested in the original problem description
    this package was constructed around (see file
    Original_problem_description.pdf in the Games_dev_tech_test
    directory)
    
    Given the set of points given in the original problem description
    in 2-dimensional Euclidean space with the given associated
    directions, finds which other points are visible for a given point
    given a maximum distance for which other points are visible and
    a maximum angle either side of the point's associated direction the
    angle of vision extends (referred to as a wedge), which other
    points are visible.
    
    Note that 'East', 'North', 'West' and 'South' are the directions
    0, 90, 180 and 270 degrees anticlockwise of the x-axis of the
    Cartesian coordinate system
    
    Additionally, this takes the result of float calculations as
    exact, so errors for points near the border of the visible region
    (particularly the wedge border) may occur.
    
    Args:
        Required positional:
        name (hashable object): The unique identifier of the
                original point
        wedge_angle (int/float): The maximum angle either side of the
                point's associated direction that other points are
                visible, in degrees
        max_dist (int/float): The chosen maximum distance from the
                point that other points are visible
        
    Returns:
    Set of hashable objects, representing the unique identifier of
    the other points that are visible from a given points for the
    given parameters
    """
    ps = getPointSet1()
    other_visible = ps.otherVisiblePoints(name, wedge_angle, max_dist)
    res = other_visible[0].union(other_visible[1])
    res.discard(name)
    return res

def visiblePointsPrintExample1() -> None:
    """
    Prints an example of the use of the function isVisible() for
    pre-selected queries, printing to console details of the
    points and the results of queries
    
    Args:
        None
    
    Returns:
    None
    """
    eps = 10 ** -5
    queries = [
        (1, 45, 20),
        (1, 135, 20),
        (1, 0, 20),
        (4, 40, 15),
    ]
    ps = getPointSet1()
    
    ps_str_lst = [
        "For the PointSet object with the following points:"
    ]
    for name, pos, direct in zip(
        ps.point_name_list,
        ps.point_positions,
        ps.point_directs
    ):
        ps_str_lst.append(
            f" The point with unique identifier {name} at position "
            f"{pos} with associated direction "
            f"{ps.direct_dict_inv[direct]}"
        )
    print("\n".join(ps_str_lst))
    print("\n\nthe following queries gave the results:\n\n")
    q_str_lst = []
    for q in queries:
        res = isVisible(*q)
        nm = ps.point_name_dict[q[0]]
        direct = ps.direct_dict_inv[ps.point_directs[nm]]
        stem = (
            f"From the point numbered {q[0]} with the visibility cone "
            f"extending {q[1]} degrees either side of the direction "
            f"{direct} and maximum distance {q[2]},"
        )
        if not res:
            q_str_lst.append(f"{stem} no other points are visible.")
            continue
        res_str = ", ".join([str(x) for x in sorted(res)])
        q_str_lst.append(
            f"{stem} the other points corresponding to the following "
            f"numbers are visible:\n {res_str}"
        )
    print("\n\n".join(q_str_lst))
    return
