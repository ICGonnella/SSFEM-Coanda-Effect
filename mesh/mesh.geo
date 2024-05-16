ll = 1.2;
lh = 1.2;

Point(1) = {50, 7.5, 0, ll};
Point(2) = {10, 7.5, 0, lh};
Point(3) = {10, 5, 0, lh};
Point(4) = {0, 5, 0, lh};
Point(5) = {0, 2.5, 0, lh};
Point(6) = {10, 2.5, 0, lh};
Point(7) = {10, 0, 0, lh};
Point(8) = {50, 0, 0, ll};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 1};

Line Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};
Plane Surface(1) = {1};

Physical Line("Inlet") = {4};
Physical Line("Outlet") = {8};
Physical Line("Wall") = {1, 2, 3, 5, 5, 7};
Physical Surface("Domain") = {1};

Recombine Surface {1};
