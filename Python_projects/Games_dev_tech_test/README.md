README.md file for the Visible_points_identification Python module

The module was written to provide a solution to the problem stated
in the file Original_problem_description.pdf in the current directory
(Games_dev_tech_test), and related, more general problems. The module
with associated code and unit tests are contained in the
directory Visible_points_identification. From this module, the
following objects and functions can be imported (see the associated
documentation for more detail):

- PointSet: class representing a set of points and associated
      directions, with methods enabling the calculation of visible
      points given a specified position and other parameters.
- isVisibleArbitraryPointsAndQueries(): Function which creates
      a PointSet object based on a given set of points and associated
      directions and answers to given queries based on that PointSet
      object. A query asks which other points in the PointSet object
      are visible from a given point in the PointSet based on
      parameters specifying the region that can be seen from that point
      (the visible cone described in the original problem description).
- isVisible(): Function which is a restricted version of
      isVisibleArbitraryPointsAndQueries(), specific to the PointSet
      object representing the points and associated directions in the
      original problem description, and only processing a single
      query per function call.
      This is the function requested by the originally stated problem
- randomPointSetCreator(): Creates a PointSet based on a specified
      number of random positions and directions.

This was written using Python 3.11 in Ubuntu 18.04. The requirements
of the module are given in requirements.txt in the current directory
(Games_dev_tech_test).

An example of the use of this module, with the results given in text
printed to console can be run using the console command:
    python -m Visible_points_identification
from within the current directory (Games_dev_tech_test). This example
illustrates the use of the isVisible function (which, as noted above,
utilises a PointSet representing the set of points and associated
directions given in the original problem statement
(Original_problem_description.pdf) with some queries, including the
example query in that statement.

Unit tests of the module can be carried out by using the console
command:
    python -m unittest discover Visible_points_identification/
from within the current directory (Games_dev_tech_test)
