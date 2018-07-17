## setcover problem on lecture 4
reset;
set nod;

param A{nod,nod};
var x{i in nod} binary;

minimize cost: sum{i in nod}x[i];

subject to cover {i in nod}: sum{j in nod}A[i,j]*x[j]>=1;

