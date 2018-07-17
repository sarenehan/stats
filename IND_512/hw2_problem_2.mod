var x1 >= 0;
var x2 >= 0;

maximize profit: 3 * x1 + 5 * x2;

subject to
c1: x1 + 2 * x2 <= 40;
c2: 2 * x1 + x2 <= 50;

option solver cplex;
option cplex_options 'sensitivity';

solve;

display x1, x2;

print 'Reduced Costs:';
display x1.rc, x2.rc;

print 'Shadow Prices:';
display c1;
display c2;

print 'Upper and Lower bound of Problem Coefficients';
display x1.up, x2.up;
display x1.down, x2.down;

print 'Upper and lower bound of right hand side constraints';
display c1.up, c2.up;
display c1.down, c2.down;
