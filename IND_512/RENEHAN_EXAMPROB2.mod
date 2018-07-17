var product1 >= 0;
var product2 >= 0;
var product3 >= 0;
var product4 >= 0;

maximize profit: (300 - (8 * 8) - (7 * 6)) * product1 +
                 (260 - (5 * 8) - (8 * 6)) * product2 +
                 (220 - (4 * 8) - (7 * 6)) * product3 +
                 (180 - (6 * 8) - (4 * 6)) * product4;


subject to
skilled_labor: 8 * product1 + 5 * product2 + 4 * product3 + 6 * product4 <= 600;
unskilled_labor: 7 * product1 + 8 * product2 + 7 * product3 + 4 * product4 <= 650;
machine_1_time: 11 * product1 + 7 * product2 + 6 * product3 + 5 * product4 <= 700;
machine_2_time: 4 * product1 + 6 * product2 + 5 * product3 + 4 * product4 <= 500;

option solver cplex;
option cplex_options 'sensitivity';

solve;

display product1, product2, product3, product4;

print 'Reduced Costs:';
display product1.rc, product2.rc, product3.rc, product4.rc;

print 'Shadow Prices:';
display skilled_labor;
display unskilled_labor;
display machine_1_time;
display machine_2_time;

print 'Upper and Lower bound of Problem Coefficients';
display product1.up, product2.up, product3.up, product4.up;
display product1.down, product2.down, product3.down, product4.down;

print 'Upper and lower bound of right hand side constraints';
display skilled_labor.up, unskilled_labor.up, machine_1_time.up, machine_2_time.up;
display skilled_labor.down, unskilled_labor.down, machine_1_time.down, machine_2_time.down;
