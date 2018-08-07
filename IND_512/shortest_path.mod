reset;
set INTER; #intersections

param entr symbolic in INTER;  #entrance to road network
param exit symbolic in INTER, <> entr; # exit from road network

set ROADS within (INTER diff {exit}) cross (INTER diff {entr});

param time {ROADS} >=0;  #time to travel roads
var Use {(i,j) in ROADS}>=0; # 1 if (i,j) in shortest path

minimize Total_Time: sum {(i,j) in ROADS} time [i,j]*Use[i,j];

subject to Start: sum {(entr,j) in ROADS} Use[entr,j]=1;

subject to Balance {k in INTER diff {entr, exit}}:
	sum{(i,k) in ROADS} Use[i,k]=sum{(k,j) in ROADS} Use [k,j];
	
data;

set INTER := 1 2 3 4 5 6 ;

param entr := 1;
param exit := 6;

param: ROADS: time  :=
      1 2      10,  1 3  25
      2 3      35,  3 4  20
      2 5      15,  3 5  35
      4 5      40,  4 6  30
      5 6      20;
      
 option solver cplex;
 solve;
 display Use;