set U={50,300,500};

param demand in U default 50;

param n_beers = 312.5;

param return = -(0.5 * n_beers) +
((1.5 * min(demand, n_beers)) + (.1 * max((n_beers - demand), 0)));

let demand:=50; solve; display n_beers, demand, return;
let demand:=300; solve; display n_beers, demand, return;
let demand:=500; solve; display n_beers, demand, return;

