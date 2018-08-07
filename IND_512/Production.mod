reset;

set period;
set periodh;

param Demand {period};
param capacity {period};

param setupCost {period};
param productCost {period};
param h {periodh};

var y{period} binary;
var x{i in period} integer , <= capacity[i];
var s{periodh} integer >=0;

minimize cost: sum{i in period}(setupCost[i]*y[i]+productCost[i]*x[i])+sum{j in periodh}s[j]*h[j];
subject to ballance{i in period}: x[i]+s[i-1]=Demand[i]+s[i];
### 100000000 is just a large number 
subject to logical{i in period}: x[i] <= y[i]*100000000;
subject to firstHolding: s[0]=0;