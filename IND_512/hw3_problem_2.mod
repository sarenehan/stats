set customer;
set warehouse;

param h{warehouse, customer};
param demand{customer};
param setupCost{warehouse};

var x{warehouse} >= 0;
var production{i in warehouse, j in customer} >= 0;

minimize cost:
    sum{i in warehouse}(setupCost[i]*x[i]+sum{j in customer}(production[i,j]*(5 + h[i,j])));

subject to
    customer_demand{j in customer}: sum{i in warehouse}production[i,j]=demand[j];
subject to
    production_limit{i in warehouse}: sum{j in customer}production[i,j] <= 70;
subject to
    exist_to_produce{i in warehouse}: sum{j in customer}production[i,j]<=x[i]*10000000;