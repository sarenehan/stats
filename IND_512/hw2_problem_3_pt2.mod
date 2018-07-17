var y1 >= 0;
var y2 >= 0;
var y3 >= 0;

minimize constraints: 600 * y1 + 400 * y2 + 500 * y3;

subject to
x1: 4 * y1 + y2 + 3 * y3 >= 6;
x2: 9 * y1 + y2 + 4 * y3 >= 10;
x3: 7 * y1 + 3 * y2 + 2 * y3 >= 9;
x4: 10 * y1 + 40 * y2 + y3 >= 20;

solve;

display y1, y2, y3;