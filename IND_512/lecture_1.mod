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

solve;

display brownie, chocolate_ice_cream, cola, pineapple_cheesecake;