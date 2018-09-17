reset;

param yield_wheat default 2.5;
param yield_corn default 3;
param yield_bean default 20;

set U={0.8,1,1.2};

param uncertainty in U default 1;

var x1 >=0; # acres of wheat
var x2 >=0; # acres of corn
var x3 >=0; # acres of beans

var w1>=0; # Sold wheat
var w2>=0; # sold corn
var w3>=0 <=6000; #sold bean
var w4>=0; #bean sold

var y1>=0; #purchase wheat
var y2>=0; #purchase corn

maximize return: 170*w1+150*w2+36*w3+10*w4
-(150*x1+230*x2+260*x3)-(238*y1+210*y2);

subject to
area: x1+x2+x3<=500;
wheat: x1=120;
corn: x2=80;
bean: x3=300;
excess_wheat: w1<=uncertainty*yield_wheat*x1-200+y1;
excess_corn: w2<=uncertainty*yield_corn*x2-240+y2;

purchased_wheat: y1>=200-uncertainty*yield_wheat*x1;
purchased_corn: y2>=240-uncertainty*yield_corn*x2;

total_bean: uncertainty*yield_bean*x3=w3+w4;

let uncertainty:=0.8; solve; display x1, x2,x3, w1,w2,w3,w4,y1,y2;
let uncertainty:=1; solve; display x1, x2,x3, w1,w2,w3,w4,y1,y2;
let uncertainty:=1.2; solve; display x1, x2,x3, w1,w2,w3,w4,y1,y2;