set terminal pngcairo enhanced font "Arial,12" size 1200,800
set output 'complexdouble.png'

set xlabel 'Matrix size (N)'
set ylabel 'Time (s)'
set title 'Hermitian matrix diagonalization (double precision)'

set logscale x
set logscale y

# legend to top left
set key top left

# Dashed grid at major ticks. But looks weird with cropped x-axis, so commented out
#set grid xtics ytics lw 0.5 lt 2 lc rgb "gray"

files = system("ls *.dat")

# find x-axis range 
xmin = 1e100
xmax = -1e100

do for [file in files] {
    stats file using 1 nooutput
    xmin = (STATS_min < xmin) ? STATS_min : xmin
    xmax = (STATS_max > xmax) ? STATS_max : xmax
}

set xrange [xmin:xmax]

plot for [file in files] file using 1:2 skip 1 with linespoints title system(sprintf("basename %s .dat", file)) noenhanced

