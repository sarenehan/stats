set cities;

param h{cities,cities};

var x{cities} binary;

minimize n_stations: sum{i in cities}x[i];

subject to distances{i in cities}:
    sum{j in cities} (if h[i,j] <= 15 then x[j] else 0) >= 1;

# Output:
# x [*] :=
# C1  0
# C2  1
# C3  0
# C4  1
# C5  0
# C6  0
# ;

# n_stations = 2