var p1 binary;
var p2 binary;
var p3 binary;
var p4 binary;
var p5 binary;
var p6 binary;
var p7 binary;
param M = 100000000;

maximize defensive_ability: 3 * p1 + 2 * p2 + 2 * p3 + 1 * p4 + 3 * p5 + 3 * p6 + 1 * p7;

subject to
five_players: p1 + p2 + p3 + p4 + p5 + p6 + p7 = 5;
four_guards: p1 + p3  + p5 + p7 >= 4;
two_forwards: p3 + p4 + p5 + p6 + p7 >= 2;
one_center: p2 + p4 + p6 >= 1;
avg_ball_handling: (p1 * 3 + p2 * 2 + p3 * 2 + p4 * 1 + p5 * 3 + p6 * 3 + p7 * 3) / 5 >= 2;
avg_shooting: (p1 * 3 + p2 * 1 + p3 * 3 + p4 * 3 + p5 * 3 + p6 * 1 + p7 * 2) / 5 >= 2;
avg_rebounding: (p1 * 1 + p2 * 3 + p3 * 2 + p4 * 3 + p5 * 3 + p6 * 2 + p7 * 2) / 5 >= 2;
c3: p3 + p6 <= 1;
c4: p4 + p5 >= 2 - (M * (1 - p1));
c5: p2 + p3 = 1;

option solver cplex;
solve;
display p1, p2, p3, p4, p5, p6, p7;
