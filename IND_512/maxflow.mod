reset;

var xs1 <=1, >=0;
var xs2 <=1, >=0;
var xs3 <=1, >=0;
var xs4 <=1, >=0;
var x5t <=1, >=0;
var x6t <=1, >=0;
var x7t <=1, >=0;
var x8t <=1, >=0;
var x16 >=0;
var x25 >=0;
var x27 >=0;
var x28 >=0;
var x36 >=0;
var x46 >=0;
var x47 >=0;
var x48 >=0;
var v;

maximize flow: v;

subject to
F: xs1+xs2+xs3+xs4=v;
F1: -xs1+x16=0;
F2: -xs2+x25+x27+x28=0;
F3: -xs3+x36=0;
F4: -xs4+x46+x47+x48=0;
F5: -x25+x5t=0;
F6: -x16-x36-x46+x6t=0;
F7: -x27-x47+x7t=0;
F8: -x28-x48+x8t=0;

option solver cplex;
solve;
display xs1;
display  xs2;
display  xs3;
display  xs4;
display  x5t;
display  x6t;
display  x7t;
display  x8t;
display  x16;
display  x25;
display  x27;
display  x28;
display  x36;
display  x46;
display  x47;
display  x48;
display  v;