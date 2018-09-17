set nodes;
param orig symbolic in nodes;
param dest symbolic in nodes, <> orig;

set arcs within {nodes, nodes};

param upcap {arcs}>=0;
param lowcap {arcs}>=0;
var flow{(i,j) in arcs} >= lowcap[i,j], <= upcap[i,j];
var v;

minimize total_flow: v;

subject to orignflow: sum{(orig,j) in arcs}flow[orig,j]=v;
subject to destflow: -sum{(i,dest) in arcs}flow[i,dest]=-v;
subject to balance {k in nodes diff {orig, dest}}:
sum{(i,k) in arcs} flow[i,k]=sum{(k,j) in arcs}flow[k,j];

data;

set nodes:= s shift1 shift2 shift3 dept1 dept2 dept3 t;

param orig:= s;
param dest:= t;

param: arcs: upcap:=
shift1 dept1 8
shift1 dept2 12
shift1 dept3 12
shift2 dept1 6
shift2 dept2 12
shift2 dept3 12
shift3 dept1 4
shift3 dept2 12
shift3 dept3 7
s shift1 32
s shift2 30
s shift3 23
dept1 t 18
dept2 t 36
dept3 t 31;

param: lowcap:=
shift1 dept1 6
shift1 dept2 11
shift1 dept3 7
shift2 dept1 4
shift2 dept2 11
shift2 dept3 7
shift3 dept1 2
shift3 dept2 10
shift3 dept3 3
s shift1 26
s shift2 24
s shift3 19
dept1 t 13
dept2 t 32
dept3 t 22;

solve;
display flow;
