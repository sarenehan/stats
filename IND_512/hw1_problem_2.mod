var lager >= 0;
var ale >= 0;

maximize revenue: 5 * lager + 2 * ale;

subject to
corn: 5 * lager + 2 * ale <= 60;
hops: 2 * lager + 1 * ale <= 25;

solve;

display lager, ale;