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

set nodes:= s job_1 job_2 job_3 job_4 t01 t12 t23 t35 t56 t67 t;

param orig:= s;
param dest:= t;

param: arcs: cap:=
s job_1 2.5
s job_2 2.1
s job_3 3.1
s job_4 1.8
job_3 t01 2
job_3 t12 2
job_3 t23 2
job_3 t35 4
job_3 t56 2
job_1 t12 2
job_1 t23 2
job_2 t56 2
job_2 t67 2
job_4 t23 2
job_4 t35 4
t01 t 2
t12 t 2
t23 t 2
t35 t 4
t56 t 2
t67 t 2;

solve;
display flow;
