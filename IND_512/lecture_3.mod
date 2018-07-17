reset;

set ROWS;
set COLS;
param A {ROWS, COLS} default 0;

var x{COLS}>=0;
var v;

maximize zot: v;

subject to ineqs {i in ROWS}:
     sum{j in COLS} -A[i,j]*x[j]+v<=0;

subject to equal:
     sum{j in COLS} x[j]=1;

