reset;
set YEAR;

param buy_machine symbolic in YEAR;
param get_rid_of_machine symbolic in YEAR, <> buy_machine;

set own within (YEAR diff {get_rid_of_machine}) cross (YEAR diff {buy_machine});

param cost {own} >=0;
var Use {(i,j) in own}>=0;

minimize total_cost: sum {(i,j) in own} cost [i,j]*Use[i,j];

subject to Start: sum {(buy_machine,j) in own} Use[buy_machine,j]=1;

subject to Balance {k in YEAR diff {buy_machine, get_rid_of_machine}}:
    sum{(i,k) in own} Use[i,k]=sum{(k,j) in own} Use [k,j];

data;

set YEAR := 1 2 3 4 5 6 ;

param buy_machine := 1;
param get_rid_of_machine := 6;

param: own: cost  :=
      1 2      208000
      1 3      258000
      1 4      355000
      1 5      537000
      1 6      841000
      2 3      228000
      2 4      278000
      2 5      375000
      2 6      557000
      3 4      248000
      3 5      298000
      3 6      395000
      4 5      288000
      4 6      338000
      5 6      338000
      ;

 option solver cplex;
 solve;
 display Use;