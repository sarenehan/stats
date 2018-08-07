reset;

set supplier;
set consumer;

param cost{supplier, consumer};
param c{supplier};
param d{consumer};

var x{supplier, consumer}>=0;

maximize transCost: sum{i in supplier, j in consumer}x[i,j]*cost[i,j];

subject to capacity {i in supplier}: sum{j in consumer}x[i,j]=c[i];

subject to demand {j in consumer}: sum{i in supplier}x[i,j]=d[j];

data;

set supplier:= w1 w2 w3;
set consumer:= HC1 HC2 HC3 HC4;

param cost: HC1 HC2 HC3 HC4:=
		  w1 464 513 654 867
		  w2 352 416 690 791
		  w3 995 682 388 685;
		  
param c:= w1 75 w2 125 w3 100;
param d:= HC1 80 HC2 65 HC3 70 HC4 85;

option solver cplex;
solve;
display x;
display transCost;