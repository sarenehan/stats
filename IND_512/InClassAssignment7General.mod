reset;

set nodes;
param orig symbolic in nodes;
param dest symbolic in nodes, <> orig;

set arcs within {nodes, nodes};

param C{arcs};
param cap {arcs}>=0;
var flow{(i,j) in arcs} >= 0, <= cap[i,j];


minimize total_cost: sum{(i,j) in arcs} flow[i,j]*C[i,j];

subject to orignflow: sum{(orig,j) in arcs}flow[orig,j]=900;
subject to destflow: -sum{(i,dest) in arcs}flow[i,dest]=-900;
subject to balance {k in nodes diff {orig, dest}}:
sum{(i,k) in arcs} flow[i,k]=sum{(k,j) in arcs}flow[k,j];

data;

set nodes:= s a b c d t ;

param orig:= s;
param dest:= t;

param: arcs: cap:=
s a 800 a c 600 b d 400 c d 600 d t 600
s b 200  b c 300      c t 400
       a d 100              ;

param: C:=
s a 10 a c 30 b d 60 c d 30 d t 30
s b 50  b c 10      c t 60
       a d 70              ;    
solve;
display total_cost;
display flow;
