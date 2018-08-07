reset;

param n;
#######
#Sets#
######
set Nodes=1..n;
set Arcs within {i in Nodes, j in Nodes: i<>j };

#####
#Parameters#
#########

param length {Arcs} ;


#############
# Variables #
#############

var x{Arcs} binary ;

#########
# Model #
#########

minimize Tourlength:
sum {(i,j) in Arcs} length[i,j]*x[i,j];



#constraints
#exactly one outgoing
subject to Degree1{i in Nodes}: sum {(i,j) in Arcs} x[i,j] =1;

#exactly one ingoing
subject to Degree2{i in Nodes}: sum {(j,i) in Arcs} x[j,i] =1;

#subtour elimination after solving without any subtour elimination
# a subtour with S={3,4} and S'={1,2,5} is create. The following
# constraint will eliminate that
subject to subtour1: x[5,4] + x[5,3] + x[2,4] + x[2,3] + x[1,4] +
x[1,3]+x[4,1]+x[4,2]+x[4,5]+x[3,1]+x[3,2]+x[3,5]>=2;

#after solving with the above subtour constraint we get subtours
# S={2,5} and S'={1,4,3}
subject to subtour2: x[4,2] + x[4,5] + x[3,2] + x[3,5] + x[1,2] +
x[1,5]+x[5,1]+x[5,3]+x[5,4]+x[2,1]+x[2,3]+x[2,4]>=2;

data;

param n:=5;

set Arcs:=  (1,2) (1,3) (1,4) (1,5)
			(2,1) (2,3) (2,4) (2,5)
			(3,1) (3,2) (3,4) (3,5)
			(4,1) (4,2) (4,3) (4,5)
			(5,1) (5,2) (5,3) (5,4);

param length:  1    2    3    4   5 :=
			1  .   132  217	 164  58
			2 132   .   290  201  79
			3 217  290   .   113  303
			4 164  201  113   .   196
			5 58	   79   303  196  .  ;

option solver cplex;
solve;

display {(i,j) in Arcs}x[i,j];
