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

Real = Union[int, float]

class PointSet:
    """
    Class representing a set of points in a real 2-dimensional
    Euclidean space with associated directions, whose instances are
    able to determine which of the points are in the set of visible
    points from an arbitrary point with a given vision cone (see below
    for the definitions of visible points and vision cone).
    
    The following definitions are used, where the vector space
    considered is a 2-dimensional real Euclidean vector space:
    - Wedge: A subset of the vector space defined by an ordered
            pair non-zero elements of the vector space as follows:
                Consider the set of vectors encountered when
                continuously rotating the first of the ordered pair of
                vectors in an anti-clockwise direction until the
                vector is a strictly positive scalar multiple of
                the second, including the start and end points.
                An element of the vector space is an element of the
                corresponding wedge if and only if it can be expressed
                as a non-negative scalar multiple of an element of
                that set.
            
            Note that non-negative scalar multiples of either of the
            ordered pair of vectors are considered to be part of the
            wedge, and if the two vectors in the ordered pair are
            parallel, then the wedge is precisely the non-negative
            scalar multiples of either of the pair.
            
            For example, with vectors expressed in Cartesian
            coordinates, for the following ordered pairs of vectors:
            - ((1, 0), (0, 1)) defines the wedge which is the set of
              vectors (x, y) for which x >= 0 and y >= 0
            - ((0, 1), (1, 0)) defines the wedge which is the set of
              vectors (x, y) for which x <= 0 or y <= 0
            - ((-1, 0), (0, -1)) defines the wedge which is the set of
              vectors (x, y) for which x >= 0 or y >= 0
            - ((1, 0), (1, 0)) defines the wedge which is the set of
              vectors (x, 0) for which x >= 0
    - Vision cone: A subset of the vector space defined by an ordered
            pair non-zero elements of the vector space and a
            non-negative real scalar value (referred to as the maximum
            distance of the vision cone) as follows:
                The intersection of the wedge defined by the ordered
                pair of elements of the vector space and the subset of
                the vector space whose Euclidean length is no greater
                than the maximum distance of the vision cone.
    - Visible points: For a given point and vision cone, the set of
            points whose position vectors can be expressed as the sum
            of the position vector of the given point and an element
            of the vision cone. For a given point and vision cone,
            points in the corresponding set of visible points are
            described as visible.
            
    
    Initialisation args:
        Optional named:
        points (list of 3-tuples): A list whose items represent the
                initial points in the PointSet object. Each entry
                should be a 3-tuple for which:
                - Index 0 contains a 2-tuple of int/float values
                  representing the position vector of the point
                  expressed in Cartesian coordinates
                - Index 1 contains a hashable object representing
                  the unique identifier of the point. Each entry in the
                  list is required to have a distinct value in this
                  entry
                - Index 2 contains a string giving the named direction
                  ('East', 'North', 'West' or 'South') associated with
                  the point
        eps (float): Strictly positive value much less than 1
                specifying the initial value of the attribute eps
            Default: class attribute eps (initially 10 ** -5)
        
    Attributes:
        point_name_list (list of hashable objects): List containing the
                unique identifiers of the points in the PointSet
                object. This defines the index of each such point in
                the PointSet object, with the index of the point
                being the index of its unique identifier in this list
                Note that the list is 0-indexed
        point_name_dict (dict whose keys and values are hashable
                objects and ints respectively): Dictionary mapping
                the name of the unique identifier of each point in the
                PointSet object to its index in the attribute
                point_name_list (and so its index in the PointSet
                object). When this and the attribute point_name_list
                are interpreted as mappings (where the former maps
                indices to names), they can be considered to be inverse
                mappings of each other
        point_positions (list of 2-tuples of ints/floats): List
                containing the position vector of each point in the
                PointSet object, with the points ordered according
                to their indices (as defined by their order in the
                attribute point_name_list), so that if the point has
                index idx in the PointSet object, then the position
                vector is contained in index idx of this attribute
        point_directs (list of floats/ints): List containing the
                direction, specified as the principal angle in degrees
                anti-clockwise from the positive x-direction associated
                with each point in the PointSet object, with the
                directions ordered according to the indices of their
                associated points (as defined by their order in the
                attribute point_name_list), so that if the point has
                index idx in the PointSet object, then its associated
                direction is contained in index idx of this attribute
        point_orders (2-tuple of SortedList objects): Gives the indices
                of the points in the PointSet object ordered in
                ascending order of their position vector's x-component
                (in index 0) and y-comopnent (in index 1). Each
                SortedList object consists of 2-tuples, whose index 0
                contains the ordering parameter (the x-component for
                the index 0 list and the y-component for the index
                1 list) and whose index 1 contains the index of the
                point in the PointSet object.
        eps (float): Strictly positive value much less than 1
                specifying the default value of eps to be used when
                identifying whether a point is visible (specifically
                with respect assessment of whether the angle of the
                point's position vector is on the borderline between
                making it visible or not visible). This may be
                (optionally) specified on initialisation by the
                argument eps, with the default value taken from the
                value of class attribute eps_def (initially 10 ** -5)
    
    Class attributes:
        eps_def (float): Strictly positive value much less than 1
                representing default value for the attribute eps
            Initial value: 10 ** -5
        direct_dict (dict whose keys are strings and values are
                ints/floats): Dictionary whose keys and corresponding
                values are the named directions and their principal
                angle values in degrees (expressed as the smallest
                non-negative angle in an anti-clockwise direction from
                the positive x-direction) respectively
            Initial value: {"East": 0, "North": 90, "West": 180,
                            "South": 270}
        direct_dict_inv (dict whose keys are ints/floats and values are
                strings): Dictionary whose keys and corresponding
                values are the angle values in degrees (expressed as
                the smallest non-negative angle in an anti-clockwise
                direction from the positive x-direction) of the named
                directions and that name respectively. When interpreted
                as mappings, this represents the inverse mapping of
                that of class attribute direct_dict respectively
            Initial value: {0: "East", 90: "North", 180: "West",
                            270: "South"}
    
    Methods:
            (see documentation of the method in question for more
            detail)
        Instance methods:
        addPoint(): Adds a point at a given position and with an
                associated name and named direction into the PointSet
                object.
        pointInVectorWedge(): For a given wedge, determines whether a
                specific position vector is inside the wedge, also
                indicating whether the point is close to the
                borderline between inclusion and exclusion in the
                wedge
        visiblePoints(): For a given point in 2-dimensional
                Euclidean space, and a specified vision cone, returns
                the unique identifiers of the points in the PointSet
                object visible or borderline visible from that point,
                partitioning the returned identifiers based on whether
                or not the calculations, when taken to be exact, found
                the associated point to be visible and whether the
                associated point is considered to be borderline
                visible.
        otherVisiblePoints(): For a point in the PointSet, and a
                specified vision cone centred on the point's associated
                direction, returns the unique identifiers of the other
                points (i.e. not including the chosen point) in the
                PointSet object visible or borderline visible from that
                point, partitioning the returned identifiers based
                on whether or not the exact calculations when taken to
                be exact, found the associated point to be visible and
                whether the associated point is considered to be
                borderline visible.
        
        Class and static methods:
        sine(): Class method calculating the sine of a given angle in
                degrees
        cosine(): Class method calculating the cosine of a given angle
                in degrees
        angleVector(): Class method calculating the normalised vector
                in 2 dimensions in the direction a given angle
                anti-clockwise from the positive x-axis
        dotProduct(): Class method calculating the dot product of two
                vectors in 2-dimensional Euclidean space
        crossProduct(): Class method calculating the cross product of
                an ordered pair of vectors in 2-dimensional Euclidean
                space
        displacementVector(): Static method calculating the
                displacement vector from one point in 2-dimensional
                Euclidean space to another
        lengthSquared(): Static method calculating the squared length
                of a vector in 2-dimensional Euclidean space
        distanceSquared(): Class method calculating the squared
                distance between two points in 2-dimensional Euclidean
                space
        calculateNormalisedBoundingBox(): Calculates the dimensions
                of a bounding box for a vision cone defined by a given
                ordered pair of normalised vectors and a maximum
                distance of 1
    """
    
    # Angles in degrees of the named directions
    # 0 degrees is in the positive x direction (i.e. East) and
    # a positive angle is anti-clockwise
    direct_dict = {"East": 0, "North": 90, "West": 180, "South": 270}
    direct_dict_inv = {y: x for x, y in direct_dict.items()}
    
    # Default epsilon value
    eps_def = 10 ** -5
    
    # Values of sine between 0 and 90 degrees inclusive that have
    # both rational angle and value
    trig_exact = {"sine": {0: 0, 30: 0.5, 90: 1}}
    
    def __init__(self,
        points: Optional[List[Tuple[Tuple[Real], Hashable, str]]]=None,
        eps: Optional[float]=None,
    ):
        self.point_positions = []
        self.point_name_list = []
        self.point_name_dict = {}
        self.point_directs = []
        self.point_orders = tuple(SortedList() for i in range(2))
        
        self.eps = eps
        
        if points is None: points = []
        for point in points:
            self.addPoint(*point)
    
    @property
    def eps(self) -> float:
        res = getattr(self, "_eps", None)
        return self.eps_def if res is None else res
    
    @eps.setter
    def eps(self, eps: float) -> None:
        self._eps = eps
    
    def addPoint(
        self,
        pos: Tuple[Real],
        name: Hashable,
        direct: str
    ) -> None:
        """
        Inserts a point into the PointSet object at a specified
        position with a specified name and associated direction.
        
        Args:
            Required positional:
            pos (2-tuple of ints/floats): Gives the Cartesian
                    coordinates representing the position of the
                    point being inserted (with the first element
                    representing the component of the displacement
                    vector from the origin in the x-direction and
                    the second the component of the displacement vector
                    in the y-direction.
            name (hashable object): The unique identifier of the
                    point being inserted (as such, is required to be
                    distinct from that of any other point in the
                    PointSet).
            direct (str): Represents the direction associated
                    with the point being inserted, where 'East' and
                    'West' represent the positive and negative
                    x-directions respectively and 'North' and 'South'
                    represent the positive and negative y-directions
                    respectively.
        
        Returns:
        None
        """
        if name in self.point_name_dict.keys():
            raise ValueError(
                f"A point with the id {name.__str__} already exists."
            )
        if direct not in self.direct_dict.keys():
            raise ValueError(
                f"'{direct}' is not a valid direction"
            )
        
        idx = len(self.point_name_list)
        self.point_name_dict[name] = idx
        self.point_name_list.append(name)
        self.point_directs.append(self.direct_dict[direct])
        
        self.point_positions.append(pos)
        self.point_orders[0].add((pos[0], idx))
        self.point_orders[1].add((pos[1], idx))
    
    @classmethod
    def sine(cls, angle: Real) -> Real:
        """
        Class method that calculates the sine of the given angle, where
        the angle is given in degrees (i.e. 360 is a full rotation).
        
        Args:
            Required positional:
            angle (int/float): The angle in degrees for which the sine
                    is to be calculated.
        
        Returns:
        Real number (int/float) representing the sine of angle.
        """
        angle %= 360
        neg, angle = divmod(angle, 180)
        if angle > 90: angle = 180 - angle
        res = cls.trig_exact["sine"].get(
            angle,
            math.sin(angle * math.pi / 180)
        )
        return -res if neg else res
    
    @classmethod
    def cosine(cls, angle: Real) -> Real:
        """
        Class method that calculates the cosine of the given angle,
        where the angle is given in degrees (i.e. 360 is a full
        rotation).
        
        Args:
            Required positional:
            angle (int/float): The angle in degrees for which the
                    cosine is to be calculated.
        
        Returns:
        Real number (int/float) representing the cosine of angle.
        """
        return cls.sine(angle + 90)
    
    @classmethod
    def angleVector(cls, angle: Real) -> Tuple[Real]:
        """
        Class method which, given an angle in degrees, finds the
        normalised vector (i.e. the vector of length 1) that is at the
        given angle in an anti-clockwise sense with relative to the
        positive x direction, expressed in Cartesian coordinates.
        
        Args:
            Required positional:
            angle (int/float): The angle in degrees at which the
                    returned vector should be pointing relative to
                    the positive x direction, where a positive angle
                    represents an anti-clockwise turn and a negative
                    angle represents a clockwise turn.
        
        Returns:
        2-tuple of ints/floats representing the described vector in
        Cartesian coordinates (i.e. giving the x- and y-components of
        the vector respectively).
        """
        return (cls.cosine(angle), cls.sine(angle))
    
    @staticmethod
    def dotProduct(vec1: Tuple[Real], vec2: Tuple[Real]) -> Real:
        """
        Static method which, given two vectors in a 2-dimensional
        vector space, both expressed in Cartesian coordinates,
        calculates the dot product (also known as the scalar product)
        of the two vectors.
        
        Args:
            Required positional:
            vec1 (2-tuple of ints/floats): The representation of the
                    first of the two vectors in Cartesian coordinates
            vec2 (2-tuple of ints/floats): The representation of the
                    second of the two vectors in Cartesian coordinates
            
        Returns:
        Real number (int/float) giving the dot product of vec1 and
        vec2.
        
        For two vectors in a 2-dimensional vector space with magnitudes
        v1 and v2 making an angle theta with each other, the dot
        product is:
            v1 * v2 * cosine(theta)
        If the representations of the two vectors in Cartesian
        coordinates are (a, b) and (c, d) respectively, then the
        dot product of the two vectors is:
            a * c + b * d
        """
        return sum(x * y for x, y in zip(vec1, vec2))
    
    @staticmethod
    def crossProduct(vec1: Tuple[Real], vec2: Tuple[Real]) -> Real:
        """
        Static method which, given two vectors in a 2-dimensional
        vector space, both expressed in Cartesian coordinates,
        calculates the cross product of the first vector with the
        second.
        Note that in 2 dimensions, the cross product is a scalar value,
        unlike in 3 dimensions in which it is a vector.
        
        Args:
            Required positional:
            vec1 (2-tuple of ints/floats): The representation of the
                    first of the two vectors in Cartesian coordinates
            vec2 (2-tuple of ints/floats): The representation of the
                    second of the two vectors in Cartesian coordinates
            
        Returns:
        Real number (int/float) giving the cross product of vec1 with
        vec2.
        
        For two vectors in a 2-dimensional vector space with magnitudes
        v1 and v2 making an angle theta with each other, the dot
        product is:
            v1 * v2 * sine(theta)
        If the representations of the two vectors in Cartesian
        coordinates are (a, b) and (c, d) respectively, then the
        cross product of the first vector with the second is:
            a * d - b * c
        Note that this operation is anti-commutative, i.e. swapping the
        order of the vectors in the cross product causes the sign to
        flip.
        """
        return vec1[0] * vec2[1] - vec1[1] * vec2[0]
    
    @staticmethod
    def displacementVector(
        pos1: Tuple[Real],
        pos2: Tuple[Real]
    ) -> Tuple[Real]:
        """
        Static method which, iven two points in a 2-dimensional vector
        space, with positions expressed in Cartesian coordinates,
        calculates the displacement vector from the first point to the
        second (i.e. the vector from the first point to the second
        point), also represented in Cartesian coordinates.
        
        Args:
            Required positional:
            pos1 (2-tuple of ints/floats): The representation of the
                    position of the first of the two points in
                    Cartesian coordinates
            pos2 (2-tuple of ints/floats): The representation of the
                    position of the second of the two points in
                    Cartesian coordinates
            
        Returns:
        2-tuple of ints/floats giving the representation of the
        displacement vector from the first point to the second
        point in Cartesian coordinates.
        """
        return tuple(y - x for x, y in zip(pos1, pos2))
    
    @staticmethod
    def lengthSquared(vec: Tuple[Real]) -> Real:
        """
        Static method which, given a vector in a 2-dimensional
        Euclidean vector space expressed in Cartesian coordinates,
        calculates the length (or equivalently, magnitude) squared.
        
        Args:
            Required positional:
            vec (2-tuple of ints/floats): The representation of the
                    vector in question in Cartesian coordinates
        
        Returns:
        Real number (int/float) giving the squared length of vec
        """
        return sum(x ** 2 for x in vec)
    
    @classmethod
    def distanceSquared(
        cls,
        pos1: Tuple[Real],
        pos2: Tuple[Real],
    ) -> Tuple[Real]:
        """
        Class method which, given two points in a 2-dimensional
        Euclidean vector space, with positions expressed in Cartesian
        coordinates, calculates the squared distance between them.
        
        Args:
            Required positional:
            pos1 (2-tuple of ints/floats): The representation of the
                    position of the first of the two points in
                    Cartesian coordinates
            pos2 (2-tuple of ints/floats): The representation of the
                    position of the second of the two points in
                    Cartesian coordinates
        
        Returns:
        Real number (int/float) giving the squared distance between
        the two points.
        """
        return cls.lengthSquared(cls.displacementVector(pos1, pos2))
    
    @classmethod
    def _pointInVectorWedgeCalculator(
        cls,
        vec1: Tuple[Real],
        vec2: Tuple[Real],
        pos: Tuple[Real],
        cp1: Real,
        cp2: Real,
    ) -> bool:
        """
        Class method which checks if, in a 2-dimensional Euclidean
        vector space, the position vector pos is in the wedge defined
        by the ordered pair of normalised vectors (vec1, vec2), where
        the vectors are expressed in Cartesian coordinates and the
        cross products of the ordered pair of vectors and the position
        vector have been pre-calculated.
        
        See the documentation of the PointSet class for a full
        definition of a wedge
        
        Args:
            Required positional:
            vec1 (2-tuple of ints/floats): The representation of the
                    first of the ordered pair of normalised (i.e.
                    length 1) vectors defining the wedge in Cartesian
                    coordinates
            vec2 (2-tuple of ints/floats): The representation of the
                    second of the ordered pair of normalised (i.e.
                    length 1) vectors defining the wedge in Cartesian
                    coordinates
            pos (2-tuple of ints/floats): The representation of the
                    position vector whose status as inside the wedge
                    it to be determined in Cartesian coordinates
            cp1 (int/float): The pre-calculated cross product of vec1
                    with pos
            cp2 (int/float): The pre-calculated cross product of vec2
                    with pos
        
        Returns:
        Boolean (bool) which is True if the position vector pos is
        assessed to be in the wedge defined by the ordered pair of
        vectors (vec1, vec2), and False otherwise.
        """
        if not cp1 and not cp2:
            return cls.dotProduct(pos, vec1) >= 0\
                or cls.dotProduct(pos, vec2) >= 0
        return (cp1 >= 0 and cp2 <= 0)\
                if (cls.crossProduct(vec1, vec2) >= 0)\
                else (cp1 >= 0 or cp2 <= 0)
    
    @classmethod
    def pointInVectorWedge(
        cls,
        vec1: Tuple[Real],
        vec2: Tuple[Real],
        pos: Tuple[Real],
        eps: Optional[float]=None,
        pos_len: Optional[Real]=None,
    ) -> Tuple[bool, bool]:
        """
        Class method which checks if, in a 2-dimensional Euclidean
        vector space, the position vector pos is in the wedge defined
        by the ordered pair of normalised vectors (vec1, vec2), where
        the vectors are expressed in Cartesian coordinates, accounting
        for potential rounding errors in float calculations by also
        indicating when the position vector direction is close to that
        of an edge of the wedge (meaning the position vector is either
        just inside or just outside the wedge), and thus is considered
        borderline in terms of its inclusion or exclusion in the wedge.
        
        See the documentation of the PointSet class for a full
        definition of a wedge
        
        Args:
            Required positional:
            vec1 (2-tuple of ints/floats): The representation of the
                    first of the ordered pair of normalised (i.e.
                    length 1) vectors defining the wedge in Cartesian
                    coordinates
            vec2 (2-tuple of ints/floats): The representation of the
                    second of the ordered pair of normalised (i.e.
                    length 1) vectors defining the wedge in Cartesian
                    coordinates
            pos (2-tuple of ints/floats): The representation of the
                    position vector whose status as inside the wedge
                    it to be determined in Cartesian coordinates
        
            Optional named:
            eps (float): A small positive real value, specifying how
                    close the direction of the position vector should
                    be to that of an edge of the wedge in order for it
                    to be labelled as borderline.
                    A position vector is labelled as borderline if and
                    only if the absolute value of the sine of the angle
                    with either of the wedge vectors is no greater than
                    this value.
                Default: class attribute eps (initially 10 ** -5)
            pos_len (int/float): The pre-calculated length of the
                    vector pos. If not specified, this is calculated
                    explicitly from pos.
                Default: Calculated directly from pos
        
        Returns:
        2-tuple of bools, where:
        - The boolean contained in index 0 indicates whether the
          position vector pos is in the wedge (when taking the float
          calculations as exact)
        - The boolean contained in index 1 indicates whether the
          position vector pos is borderline, i.e. its direction is
          close to that of an edge of the wedge, as judged by the
          absolute value of the sine of the angle between
          pos and one of the vectors defining the wedge being no
          greater than eps.
        """
        if pos_len is None:
            pos_len = math.sqrt(cls.lengthSquared(pos))
        if eps is None:
            eps = cls.eps
        eps2 = eps * pos_len
        cp1 = cls.crossProduct(vec1, pos)
        cp2 = cls.crossProduct(vec2, pos)
        
        borderline = True
        for vec, cp in ((vec1, cp1), (vec2, cp2)):
            if abs(cp) > eps: continue
            dp = cls.dotProduct(vec, pos)
            if dp > 0: break
        else: borderline = False
        return (
            cls._pointInVectorWedgeCalculator(vec1, vec2, pos, cp1, cp2),
            borderline
        )
    
    @classmethod
    def calculateNormalisedBoundingBox(
        cls,
        vec1: Tuple[Real],
        vec2: Tuple[Real],
        eps: Optional[float]=None,
    ) -> Tuple[Tuple[Real, Real], Tuple[Real, Real]]:
        """
        Class method identifying a bounding box for the vision cone
        based on the wedge defined by the ordered pair of vectors vec1
        and vec2 (expressed in Cartesian coordinates) and a maximum
        distance of 1.
        The bounding box is a closed rectangular region in the
        space with edges parallel to the x and y axes, and contains
        the all points in the vision cone and all points that are
        borderline with respect to the vision cone for the given
        value of eps (i.e. all points at a distance no greater than
        1 of the origin and whose position vectors make an angle
        with one of the vectors whose sine value is no greater than
        eps).
        This is intended to make the area of the bounding box as
        small as possible given the constraints, though the
        bounding box does not necessarily achieve the absolute
        minimum possible area.
        
        See the documentation of the PointSet class for a full
        definition of a wedge and a vision cone
        
        Args:
            Required positional:
            vec1 (2-tuple of ints/floats): The representation of the
                    first of the ordered pair of normalised (i.e.
                    length 1) vectors defining the wedge on which the
                    vision cone being bounded is based in Cartesian
                    coordinates
            vec2 (2-tuple of ints/floats): The representation of the
                    second of the ordered pair of normalised (i.e.
                    length 1) vectors defining the wedge on which the
                    vision cone being bounded is based in Cartesian
                    coordinates
            
            Optional named:
            eps (float): A small positive real value, specifying how
                    close the direction of the position vector should
                    be to that of an edge of the wedge in order for it
                    to be labelled as borderline.
                    A position vector is labelled as borderline if and
                    only if the absolute value of the sine of the angle
                    with either of the wedge vectors is no greater than
                    this value.
                Default: class attribute eps (initially 10 ** -5)
        
        Returns:
        2-tuple of 2-tuples of ints/floats. The tuples in indices 0 and
        1 represent the x values and y values respectively of the
        edges of the bounding box perpendicular to the corresponding
        axis in increasing order.
        Note that the bounding box is closed (i.e. the edges are
        included)
        """
        # Assumes vec1 and vec2 are normalised (i.e. have length 1)
        if eps is None:
            eps = cls.eps
        bb = []
        basis_vec = [0, 0]
        for i, (cp1_prov, cp2_prov) in\
                enumerate(((-vec1[1], -vec2[1]), (vec1[0], vec2[0]))):
            bb.append([])
            for j in (-1, 1):
                basis_vec[i] = j
                if cls._pointInVectorWedgeCalculator(
                    vec1,
                    vec2,
                    basis_vec,
                    cp1_prov * j,
                    cp2_prov * j
                ):
                    bb[-1].append(1)
                else:
                    bb[-1].append(max(0, min(1,\
                            max(vec1[i] * j, vec2[i] * j) + eps)))
                bb[-1][-1] *= j
            bb[-1] = tuple(bb[-1])
            basis_vec[i] = 0
        return tuple(bb)
    
    def _filterPoints(
        self,
        bounding_box: Tuple[Tuple[Real, Real], Tuple[Real, Real]],
        eps: Optional[float]=None,
    ) -> Set[int]:
        """
        Method that uses a bounding of visible points for a given
        position and vision cone to identify the points in the PointSet
        that may be visible points (including borderline visible
        points).
        
        See the documentation of the PointSet class for a full
        definition of a wedge, a vision cone and visible points
        
        Args:
            Required positional:
            bounding_box (2-tuple of 2-tuples of ints/floats): A
                    closed rectangular region with edges parallel to
                    the x and y axes in which the whole set of visible
                    points in question (including borderline visible
                    points for the given value of eps) are guaranteed
                    to be.
                    The tuples in indices 0 and 1 represent the x
                    values and y values respectively of the edges of
                    the bounding box perpendicular to the corresponding
                    axis in increasing order.
                    This is intended to be calculated based on the
                    normalised bounding box found using the class
                    method calculateNormalisedBoundingBox(), scaling
                    and translating the bounding box by the maximum
                    distance of the vision cone and the position
                    respectively.
            
            Optional named:
            eps (float): A small positive real value, specifying how
                    close the direction of the position vector should
                    be to that of an edge of the wedge in order for it
                    to be labelled as borderline.
                    A position vector is labelled as borderline if and
                    only if the absolute value of the sine of the angle
                    with either of the wedge vectors is no greater than
                    this value.
                Default: class attribute eps (initially 10 ** -5)
        
        Returns:
        Set of ints, representing the indices of the points in the
        PointSet object inside the bounding box and so are possibly
        visible points for the position and vision cone on which the
        bounding box is based.
        """
        n = len(self.point_positions)
        curr_pnts = set(range(n))
        for j, (pnt_order, rng) in\
                enumerate(zip(self.point_orders, bounding_box)):
            i1 = pnt_order.bisect_left((rng[0], -float("inf")))
            i2 = pnt_order.bisect_right((rng[1], float("inf")))
            n_internal = (i2 - i1)
            inspect_internal = (n_internal << 1) <= n
            n_inspect = n_internal if inspect_internal\
                    else n - n_internal
            if n_inspect > len(curr_pnts):
                prev_pnts = curr_pnts
                curr_pnts = set()
                for pnt in prev_pnts:
                    pos = self.point_positions[pnt][j]
                    if rng[0] <= pos <= rng[1]:
                        curr_pnts.add(pnt)
            elif inspect_internal:
                prev_pnts = curr_pnts
                curr_pnts = set()
                for i in range(i1, i2):
                    idx = pnt_order[i][1]
                    if idx in prev_pnts:
                        curr_pnts.add(idx)
            else:
                for order_iter in (range(i1), range(i2, n)):
                    for i in order_iter:
                        idx = pnt_order[i][1]
                        curr_pnts.discard(idx)
            if not curr_pnts: return curr_pnts
        return curr_pnts
    
    def visiblePoints(
        self,
        pos: Tuple[Real],
        angle1: Real,
        angle2: Real,
        max_dist: Real,
        eps: Optional[float]=None,
    ) -> Tuple[Set[Hashable], Set[Hashable], Set[Hashable]]:
        """
        Given a point (that is not required to belong to the
        PointSet), an ordered pair of angles (both expressed in
        degrees) and a maximum distance, returns the unique identifiers
        of the points in the PointSet that are visible, i.e. the points
        for which the displacement vector from the chosen point both:
         1- has a length no greater than the chosen maximum distance
         2- has an angle (the principal angle plus an integer multiple
            of 360) from the positive x-direction in an anti-clockwise
            sense no less than the first of the pair of angles and no
            greater than the second, with a tolerance for the sine
            of the angle difference of eps.
        The returned points are partitioned as follows:
        - The points that are visible and not close to the edges of
          the wedge (non-borderline visible points)
        - The points that are visible (based on the calculations
          being taken as exact) but are close to one of the edges of
          the wedge (the borderline visible points)
        - The points that are not visible but are at a distance no
          greater than max_dist of pos and are close to one of the
          edges of the wedge (the borderline non-visible points)
        The assessment of whether a point is close to one of the wedge
        edges is based on the parameter eps (see Args entry eps for
        details)
        
        Args:
            Required positional:
            pos (2-tuple of ints/floats): The representation of the
                    position of the chosen point
            angle1 (int/float): The first of the ordered pair of
                    angles, expressed in degrees
            angle2 (int/float): The second of the ordered pair of
                    angles, expressed in degrees
            max_dist (int/float): The chosen maximum distance
            
            Optional named:
            eps (float): A small positive real value, specifying how
                    close the direction of the position vector should
                    be to that of an edge of the wedge in order for it
                    to be labelled as borderline.
                    A position vector is labelled as borderline if and
                    only if the absolute value of the sine of the angle
                    with either of the wedge vectors is no greater than
                    this value.
                Default: class attribute eps (initially 10 ** -5)
        
        Returns:
        3-tuple of sets of hashable objects, where:
        - The set contained at index 0 represents the
          non-borderline visible points
        - The set contained at index 1 represents the borderline
          visible points
        - The set contained at index 2 represents the borderline
          non-visible points
        """
        if eps is None:
            eps = self.eps
        
        if max_dist < 0 or angle1 > angle2: return []
        angle_restricted = False
        if angle2 - angle1 >= 360:
            bounding_box_norm = ((-1, 1), (-1, 1))
        else:
            angle_restricted = True
            vec1, vec2 = tuple(map(self.angleVector, (angle1, angle2)))
            bounding_box_norm = self.calculateNormalisedBoundingBox(
                    vec1, vec2, eps=eps)
        bounding_box = tuple(tuple(x + y * max_dist for y in rng)\
                for x, rng in zip(pos, bounding_box_norm))
        
        n = len(self.point_positions)
        filtered_pnts = self._filterPoints(bounding_box, eps=eps)
        if not filtered_pnts: return []
        
        res = (set(), set(), set())
        dist_sq = max_dist ** 2
        if not angle_restricted:
            for pnt in filtered_pnts:
                pos2 = self.point_positions[pnt]
                if self.distanceSquared(pos, pos2) <= dist_sq:
                    res[0].add(self.point_name_list[pnt])
            return res
        for pnt in filtered_pnts:
            pos2 = self.point_positions[pnt]
            displ_vec = self.displacementVector(pos, pos2)
            len_sq = self.lengthSquared(displ_vec)
            if len_sq > dist_sq: continue
            in_wedge, borderline = self.pointInVectorWedge(\
                    vec1, vec2, displ_vec, eps=eps,\
                    pos_len=math.sqrt(len_sq))
            if in_wedge: idx = int(borderline)
            elif not borderline: continue
            else: idx = 2
            res[idx].add(self.point_name_list[pnt])
            
        return res

    def otherVisiblePoints(
        self,
        name: Hashable,
        wedge_angle: Real,
        max_dist: Real,
        eps: Optional[float]=None,
    ) -> Tuple[List[Hashable], List[Hashable]]:
        """
        Given a point in the PointSet (which we refer to as the
        original point), a wedge angle (expressed in degrees) and a
        maximum distance, returns the unique identifiers of the points
        in the PointSet that are visible excluding the original point,
        i.e. the points for which the displacement vector from the
        chosen point:
         1- has a length no greater than the chosen maximum distance
         2- has an angle (the principal angle plus an integer multiple
            of 360) from the positive x-direction in an anti-clockwise
            sense that is up to and including the wedge angle either
            side of the direction associated with the orignal point
            in the PointSet, with a tolerance for the sine of the angle
            difference of eps.
        The returned points are partitioned as follows:
        - The points that are visible and not close to the edges of
          the wedge (non-borderline visible points)
        - The points that are visible (based on the calculations
          being taken as exact) but are close to one of the edges of
          the wedge (the borderline visible points)
        - The points that are not visible but are at a distance no
          greater than max_dist of pos and are close to one of the
          edges of the wedge (the borderline non-visible points)
        The assessment of whether a point is close to one of the wedge
        edges is based on the parameter eps (see Args entry eps for
        details)
        
        Args:
            Required positional:
            name (hashable object): The unique identifier of the
                    original point
            wedge_angle (int/float): The wedge angle described in
                    degrees
            max_dist (int/float): The chosen maximum distance
            
            Optional named:
            eps (float): A small positive real value, specifying how
                    close the direction of the position vector should
                    be to that of an edge of the wedge in order for it
                    to be labelled as borderline.
                    A position vector is labelled as borderline if and
                    only if the absolute value of the sine of the angle
                    with either of the wedge vectors is no greater than
                    this value.
                Default: class attribute eps (initially 10 ** -5)
        
        Returns:
        3-tuple of sets of hashable objects, where:
        - The set contained at index 0 represents the
          non-borderline visible points
        - The set contained at index 1 represents the borderline
          visible points
        - The set contained at index 2 represents the borderline
          non-visible points
        """
        if eps is None:
            eps = self.eps
        idx = self.point_name_dict.get(name, None)
        if idx is None:
            raise ValueError(
                f"There is not point with the name {name.__str__}."
            )
        pos = self.point_positions[idx]
        angle0 = self.point_directs[idx]
        angle1, angle2 = angle0 - wedge_angle, angle0 + wedge_angle
        return self.visiblePoints(pos, angle1, angle2, max_dist,\
                eps=eps)

def isVisibleArbitraryPointsAndQueries(
    points: Optional[Tuple[Tuple[Real], Hashable, str]],
    queries: List[Tuple[Hashable, Real, Real]],
    eps: Optional[float]=None,
) -> List[Tuple[Set[Hashable], Set[Hashable], Set[Hashable]]]:
    """
    Given a set of points in 2-dimensional Euclidean space with
    associated directions, finds the result of a list of queries
    relating to those points.
    A query specifies one of the points and asks from the location
    of that point, given a maximum distance at which another point
    is visible and a maximum angle either side of the point's
    associated direction the angle of vision extends (referred to
    as a wedge), which other points are visible.
    The points returned for each query points are partitioned as
    follows:
    - The points that are visible and whose angle is not close to
      either of the edges of the wedge (non-borderline visible
      points)
    - The points that are visible (based on the calculations
      being taken as exact) but are close to one of the edges of
      the wedge (the borderline visible points)
    - The points that are not visible but are at a distance no
      greater than maximum distance from the original point and are
      close to one of the edges of the wedge (the borderline
      non-visible points)
    The assessment of whether a point is close to one of the wedge
    edges is based on the parameter eps (see Args entry eps for
    details)
    
    Note that 'East', 'North', 'West' and 'South' are the directions
    0, 90, 180 and 270 degrees anticlockwise of the x-axis of the
    Cartesian coordinate system
    
    Args:
        Required positional:
        points (list of 3-tuples): A list whose items represent the
                initial points in question. Each entry should be a
                3-tuple for which:
                - Index 0 contains a 2-tuple of int/float values
                  representing the position vector of the point
                  expressed in Cartesian coordinates
                - Index 1 contains a hashable object representing
                  the unique identifier of the point. Each entry in the
                  list is required to have a distinct value in this
                  entry
                - Index 2 contains a string giving the named direction
                  ('East', 'North', 'West' or 'South') associated with
                  the point
        query (list of 3-tuples): A list whose items represet the
                queries made of the points. Each entry should be
                a 3-tuple for which:
                - Index 0 contains a hashable object giving the unique
                  identifier of the point from which vision is to
                  originate
                - Index 1 contains an angle, representing the maximum
                  angle in degrees either side of the direction
                  associated with the selected point that other points
                  can be considered visible
                - Index 2 contains the maximum distance at which other
                  points can be considered visible
        
        Optional named:
        eps (float): A small positive real value, specifying how
                close the direction of the position vector should
                be to that of an edge of the wedge in order for it
                to be labelled as borderline.
                A position vector is labelled as borderline if and
                only if the absolute value of the sine of the angle
                with either of the wedge vectors is no greater than
                this value.
            Default: class attribute eps (initially 10 ** -5)
    
    Returns:
    List, where the entries contain the result of the
    corresponding query (i.e. the query with the same index in
    the argument queries. Each entry is a 3-tuple of sets of hashable
    objects where for the corrseponding query:
    - The set contained at index 0 represents the
      non-borderline visible points
    - The set contained at index 1 represents the borderline
      visible points
    - The set contained at index 2 represents the borderline
      non-visible points
    """
    ps = PointSet(points, eps=eps)
    res = []
    for q in queries:
        other_visible = ps.otherVisiblePoints(*q)
        for st in other_visible:
            st.discard(q[0])
        res.append(other_visible)
    return res
