reset;

set ROWS;
set COLS;
param A {ROWS, COLS} default 0;

var x{COLS}>=0;
var v;

minimize zot: v;

subject to ineqs {j in COLS}:
     sum{i in ROWS} -A[i,j]*x[i]+v>=0;

subject to equal:
     sum{j in ROWS} x[j]=1;

