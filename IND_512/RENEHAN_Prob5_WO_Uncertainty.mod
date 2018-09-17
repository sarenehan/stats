set U={50,300,500};

param demand in U default 1;

var n_beers;

maximize return: -(0.5 * n_beers)
+ ((1.5 * min(demand, n_beers)) + (.1 * max((n_beers - demand), 0)));

subject to
fiats_are_small: n_beers<=500;

let demand:=50; solve; display n_beers, return;
let demand:=300; solve; display n_beers, return;
let demand:=500; solve; display n_beers, return;