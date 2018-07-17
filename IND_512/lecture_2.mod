# Initialize Variables
var brownie >= 0;
var chocolate_ice_cream >= 0;
var cola >= 0;
var pineapple_cheesecake >= 0;

minimize cost: 0.5 * brownie + 0.2 * chocolate_ice_cream + 0.3 * cola + 0.8 * pineapple_cheesecake;

subject to
Calories: 400 * brownie + 200 * chocolate_ice_cream + 150 * cola + 500 * pineapple_cheesecake >= 500;
Chocolate: 3 * brownie + 2 * chocolate_ice_cream >= 6;
Sugar: 2 * brownie + 2 * chocolate_ice_cream + 4 * cola + 4 * pineapple_cheesecake >= 10;
Fat: 2 * brownie + 4 * chocolate_ice_cream + cola + 5 * pineapple_cheesecake >= 8;

option solver cplex;
option cplex_options 'sensitivity';

solve;

display brownie, chocolate_ice_cream, cola, pineapple_cheesecake;

print 'Reduced Costs:';
display brownie.rc, chocolate_ice_cream.rc, cola.rc, pineapple_cheesecake.rc;

print 'Shadow Prices:';
display Calories;
display Chocolate;
display Sugar;
display Fat;

print 'Upper and Lower bound of Problem Coefficients';
display brownie.up, chocolate_ice_cream.up, cola.up, pineapple_cheesecake.up;
display brownie.down, chocolate_ice_cream.down, cola.down, pineapple_cheesecake.down;

print 'Upper and lower bound of right hand side constraints';
display Calories.up, Chocolate.up, Sugar.up, Fat.up;
display Calories.down, Chocolate.down, Sugar.down, Fat.down;