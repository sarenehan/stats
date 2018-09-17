set U={50,300,500};

param probability {U};


var n_beers;

maximize return: -(0.5 * n_beers)
+ sum {u in U} probability [u] * (
    (1.5 * min(u, n_beers)) + (.1 * max((n_beers - u), 0)));

subject to
fiats_are_small: n_beers<=500;

data;
param probability:=
50 0.15
300 0.6
500 0.25;
solve; display n_beers, return;
