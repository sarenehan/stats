var wine integer >= 0;
var beer integer >= 0;
var champagne integer >= 0;
var whiskey integer >= 0;

maximize revenue: 6 * wine + 10 * beer + 9 * champagne + 20 * whiskey;

subject to
molding: 4 * wine + 9 * beer + 7 * champagne + 10 * whiskey <= 600;
packaging: wine + beer + 3 * champagne + 40 * whiskey <= 400;
glass: 3 * wine + 4 * beer + 2 * champagne + whiskey <= 500;

option solver cplex;
solve;

display wine, beer, champagne, whiskey;
