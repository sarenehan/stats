var x1 >= 0;
var x2 >= 0;

maximize z: x1 + x2;

subject to
c1: 2 * x1 + x2 <= 4;
c2: x1 + 2 * x2 <= 3;

option solver cplex;

solve;

display x1, x2;

# Output:
# CPLEX 12.8.0.0: optimal solution; objective 2.333333333
# 2 dual simplex iterations (1 in phase I)
# x1 = 1.66667
# x2 = 0.666667