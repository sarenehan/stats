var slugger >= 0;
var eash_out >= 0;
var pct_slugger_sugar >= 0;
var pct_slugger_chocolate >= 0;
var pct_slugger_nuts >= 0;
var pct_easy_out_sugar >= 0;
var pct_easy_out_chocolate >= 0;
var pct_easy_out_nuts >= 0;

maximize revenue: 0.2 * slugger + 0.25 * eash_out;

subject to
full_slugger: pct_slugger_sugar + pct_slugger_chocolate + pct_slugger_nuts = 1;
full_easy_out: pct_easy_out_sugar + pct_easy_out_chocolate + pct_easy_out_nuts = 1;
sugar: slugger * pct_slugger_sugar + eash_out * pct_easy_out_sugar <= 100;
nuts: slugger * pct_slugger_nuts + eash_out * pct_easy_out_nuts <= 20;
chocolate: slugger * pct_slugger_chocolate + eash_out * pct_easy_out_chocolate <= 30;
min_nuts_slugger: pct_slugger_nuts >= 0.1;
min_chocolate_slugger: pct_slugger_chocolate >= 0.1;
min_nuts_easy_out: pct_easy_out_nuts >= 0.2;

solve;

display slugger, eash_out, pct_slugger_sugar, pct_slugger_chocolate, pct_slugger_nuts, pct_easy_out_sugar, pct_easy_out_chocolate, pct_easy_out_nuts;