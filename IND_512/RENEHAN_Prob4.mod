set YEAR;

param buy_phone symbolic in YEAR;
param get_rid_of_phone symbolic in YEAR, <> buy_phone;

set own within (YEAR diff {get_rid_of_phone}) cross (YEAR diff {buy_phone});

param cost {own} >=0;
var Use {(i,j) in own}>=0;

minimize total_cost: sum {(i,j) in own} cost [i,j]*Use[i,j];

subject to Start: sum {(buy_phone,j) in own} Use[buy_phone,j]=1;

subject to Balance {k in YEAR diff {buy_phone, get_rid_of_phone}}:
    sum{(i,k) in own} Use[i,k]=sum{(k,j) in own} Use [k,j];

data;

set YEAR := 1 2 3 4 5 6 7;

param buy_phone := 1;
param get_rid_of_phone := 7;

param: own: cost  :=
      1 2      60
      1 3      90
      1 4      130
      1 5      190
      1 6      260
      2 3      60
      2 4      90
      2 5      130
      2 6      190
      2 7      260
      3 4      60
      3 5      90
      3 6      130
      3 7      190
      4 5      60
      4 6      90
      4 7      130
      5 6      60
      5 7      90
      6 7      60
      ;

 option solver cplex;
 solve;
 display Use;