reset;

set projects;
set resources;

param b{resources};
param h{resources,projects};
param p{projects};


var x{projects} binary;


maximize profit: sum{i in projects}x[i]*p[i];

subject to ResLimit {j in resources}: sum{i in projects}h[j,i]*x[i]<=b[j];

