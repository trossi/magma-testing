#!/bin/bash

outdir="trimmed"
mkdir -p $outdir

for file in *.out; do
    outfile="$outdir/${file%.out}.dat"
    echo "File $file"
    awk '/= SUMMARY =/{flag=1; next} flag' "$file" > "$outfile"
    
done
