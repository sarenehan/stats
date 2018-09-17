reset;
set INTER; #intersections

param entr symbolic in INTER;  #entrance to road network
param exit symbolic in INTER, <> entr; # exit from road network

set ROADS within (INTER diff {exit}) cross (INTER diff {entr});

param max_altitude {ROADS} >=0;
var Use {(i,j) in ROADS} binary; # 1 if (i,j) in shortest path

var v;

minimize Total_max_altitude: v;
subject to ineqs{(i,j) in ROADS}: max_altitude[i,j]*Use[i,j] - v <= 0;

subject to Start: sum {(entr,j) in ROADS} Use[entr,j]=1;

subject to Balance {k in INTER diff {entr, exit}}:
    sum{(i,k) in ROADS} Use[i,k]=sum{(k,j) in ROADS} Use [k,j];

data;

set INTER := 1 2 3 4 5 6 7 8 9 10 11 12;

param entr := 1;
param exit := 12;

param: ROADS: max_altitude  :=
      1 2      1
      1 4      4
      2 3      9
      2 5      4
      4 5      3
      3 6      5
      5 6      7
      4 7      6
      5 8      5
      7 8      7
      6 9      3
      8 9      2
      7 10     6
      8 11     2
      10 11    1
      9 12     1
      11 12    2;

 option solver cplex;
 solve;
 display Use;