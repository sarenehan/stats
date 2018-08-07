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

set nodes:= s a b c d t;

param orig:= s;
param dest:= t;

param: arcs: cap:=
s a 10 a c 20 b d 35 c d 40 d t 20
s b 25 a b 35        c t 30
       a d 15              ;
    
solve;
display flow;
