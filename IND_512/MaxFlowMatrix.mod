reset;

set nodes;
param orig symbolic in nodes;
param dest symbolic in nodes, <> orig;

set arcs within {nodes, nodes};

param upcap {arcs}>=0;
param lowcap {arcs}>=0;
var flow{(i,j) in arcs} >= lowcap[i,j], <= upcap[i,j];
var v;

maximize total_flow: v;

subject to orignflow: sum{(orig,j) in arcs}flow[orig,j]=v;
subject to destflow: -sum{(i,dest) in arcs}flow[i,dest]=-v;
subject to balance {k in nodes diff {orig, dest}}:
sum{(i,k) in arcs} flow[i,k]=sum{(k,j) in arcs}flow[k,j];

data;

set nodes:= s a b c d e f t;

param orig:= s;
param dest:= t;

param: arcs: upcap:=
s a 20 a d 5 b d 7 c d 4 d t 14
s b 20 a e 7 b e 9 c e 2 e t 17
s c 13 a f 10 b f 5 c f 8 f t 22;          

param: lowcap:=
s a 19 a d 4 b d 6 c d 3 d t 13
s b 19 a e 6 b e 8 c e 1 e t 16
s c 12 a f 9 b f 4 c f 7 f t 21;  
    
solve;
display flow;
