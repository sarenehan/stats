reset;

set nodes;
param orig symbolic in nodes;
param dest symbolic in nodes, <> orig;

set arcs within {nodes, nodes};

param cap {arcs}>=0;
var flow{(i,j) in arcs} >= 0, <= cap[i,j];
var v;

maximize total_flow: v;

subject to orignflow: sum{(orig,j) in arcs}flow[orig,j]=v;
subject to destflow: -sum{(i,dest) in arcs}flow[i,dest]=-v;
subject to balance {k in nodes diff {orig, dest}}:
sum{(i,k) in arcs} flow[i,k]=sum{(k,j) in arcs}flow[k,j];

data;

set nodes:= s0 n1 n2 n3 si;

param orig:= s0;
param dest:= si;

param: arcs: cap:=
s0 n1 2
s0 n2 3
n1 n2 3
n1 n3 4
n2 si 2
n3 si 1
;

solve;
display flow;
