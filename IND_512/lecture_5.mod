set items;

param v{items};
param r{items};
var x{items} binary;


maximize value: sum{i in items}x[i]*v[i];

subject to VolumeLimit: sum{j in items}r[j] * x[j] <= 1100;