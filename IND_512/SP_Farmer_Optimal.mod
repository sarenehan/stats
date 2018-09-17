reset;
set U={0.8,1,1.2};

param yield_wheat {u in U}:=2.5*u;
param yield_corn {u in U}:=3*u;
param yield_bean {u in U}:=20*u;

param probability {U};


var x1; # acres of wheat
var x2; # acres of corn
var x3; # acres of beans

var w1 {U}>=0; # Sold wheat
var w2 {U}>=0; # sold corn
var w3 {U}>=0 <=6000; #sold bean
var w4 {U}>=0; #bean sold

var y1 {U}>=0; #purchase wheat
var y2 {U}>=0; #purchase corn

maximize return: -(150*x1+230*x2+260*x3)
+ sum {u in U} probability [u] * (170*w1[u]+150*w2[u]+36*w3[u]
+10*w4[u]-(238*y1[u]+210*y2[u]));

subject to
area: x1+x2+x3<=500;
excess_wheat {u in U}: w1[u]<=yield_wheat[u]*x1-200+y1[u];
excess_corn {u in U}: w2[u]<=yield_corn[u]*x2-240+y2[u];

purchased_wheat {u in U}: y1[u]>=200-yield_wheat[u]*x1;
purchased_corn {u in U}: y2[u]>=240-yield_corn[u]*x2;

total_bean {u in U}: yield_bean[u]*x3=w3[u]+w4[u];

data; 
param probability:=
0.8 0.333333333
1 0.333333333
1.2 0.33333333;
solve; display x1, x2,x3, w1,w2,w3,w4,y1,y2;
