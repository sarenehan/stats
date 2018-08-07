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

# subtour elimination after solving without any subtour elimination
# a subtour with S={3,5,6} and S'={1,2,4} is create. The following
# constraint will eliminate that
subject to subtour1: x[1,3] + x[1,5] + x[1,6] + x[2,3] + x[2,5] +
x[2,6]+x[4,3]+x[4,5]+x[4,6]+x[3,1]+x[3,2]+x[3,4]+ x[5,1] + x[5, 2] +
x[5, 4] + x[6, 1] + x[6, 2] + x[6,4]>=2;

#after solving with the above subtour constraint we get subtours
# S={3, 6} and S'={1,2,4,5}
subject to subtour2: x[3,1] + x[3,2] + x[3,4] + x[3,5] + x[6,1] +
x[6,2]+x[6,4]+x[6,5]+x[1,3]+x[1,6]+x[2,3]+x[2,6] + x[4,3] + x[4,6] +
x[5,3] + x[5,6]>=2;

data;

param n:=6;

set Arcs:=  (1,2) (1,3) (1,4) (1,5) (1, 6)
            (2,1) (2,3) (2,4) (2,5) (2, 6)
            (3,1) (3,2) (3,4) (3,5) (3, 6)
            (4,1) (4,2) (4,3) (4,5) (4, 6)
            (5,1) (5,2) (5,3) (5,4) (5, 6)
            (6,1) (6,2) (6,3) (6,4) (6, 5);

param length:  1    2    3    4   5   6 :=
            1  .   27   43   16  30  26
            2  7   .    16    1  30  25
            3 20   13   .    35   5   0
            4 21   16   25   .   18  18
            5 12   46   27   48   .   5
            6 12    5    5    9    5  .  ;

option solver cplex;
solve;

display {(i,j) in Arcs}x[i,j];
