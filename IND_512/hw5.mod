reset;

set plant;
set distribution_center;

param cost{plant, distribution_center};
param c{plant};
param d{distribution_center};

var x{plant, distribution_center}>=0;

minimize transCost: sum{i in plant, j in distribution_center}x[i,j]*cost[i,j];

subject to capacity {i in plant}: sum{j in distribution_center}x[i,j]=c[i];

subject to demand {j in distribution_center}: sum{i in plant}x[i,j]=d[j];

data;

set plant:= cleveland chicago boston;
set distribution_center:= dallas atlanta san_fran philly;

param cost: dallas atlanta san_fran philly:=
  cleveland      8       6       10      9
  chicago        9      12       13      7
  boston        14       9       16      5;

param c:= cleveland 35 chicago 50 boston 40;
param d:= dallas 45 atlanta 20 san_fran 30 philly 30;

option solver cplex;
solve;
display x;
display transCost;