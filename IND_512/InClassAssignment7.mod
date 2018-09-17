reset;

var x12 <= 800, >=0;
var x13<= 600, >=0;
var x25<= 100, >=0;
var x24<= 600, >=0;
var x56<= 600, >=0;
var x45<= 600, >=0;
var x46<= 400, >=0;
var x35<= 400, >=0;
var x34<= 300, >=0;

minimize z: 10*x12+50*x13+70*x25+30*x24+30*x56+30*x45+60*x35+60*x46+10*x34;

subject to
outflow: x12+x13=900;
b1: -x12+x25+x24=0;
b2: -x13+x35+x34=0;
b3: -x24+x45+x46-x35-x34=0;
b4: -x25+x56-x45-x35=0;
b5: -x56-x46=-900;

option solver cplex;

solve;

display z;
display x12, x13, x25, x24, x56, x45, x46, x35, x34;
