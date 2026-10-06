#!/bin/sh
# Batch-runs imageColourNPP over a folder of photos and keeps evidence of the run:
#   <out>/source/   the original photos           (<name>_before.jpg)
#   <out>/results/  colour-splash and invert output (<name>_after_colour.jpg, <name>_after_invert.jpg)
#   <out>/logs/     one combined run log, per-image logs and a CSV summary
#
# Usage: ./run.sh [input_dir] [output_dir] [count]
#   defaults: D:/images/models/charlize  data/charlize  100

IN_DIR=${1:-D:/images/models/charlize}
OUT_DIR=${2:-data/charlize}
COUNT=${3:-100}
EXE=bin/imageColourNPP
[ -f "$EXE.exe" ] && EXE="$EXE.exe"

mkdir -p "$OUT_DIR/source" "$OUT_DIR/results" "$OUT_DIR/logs"
LOG="$OUT_DIR/logs/run.log"
CSV="$OUT_DIR/logs/summary.csv"
echo "image,mode,width,height,common_hue_deg,kept_percent,seconds,status" > "$CSV"

{
    echo "=== imageColourNPP batch run ==="
    echo "date:   $(date)"
    echo "input:  $IN_DIR (first $COUNT .jpg files)"
    echo "output: $OUT_DIR"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/gpu:    /'
    nvcc --version 2>/dev/null | tail -2 | sed 's/^/nvcc:   /'
    echo
} | tee "$LOG"

start_all=$(date +%s)
n=0; failed=0
for src in $(ls "$IN_DIR"/*.jpg | head -n "$COUNT"); do
    n=$((n + 1))
    name=$(basename "$src" .jpg)
    cp "$src" "$OUT_DIR/source/${name}_before.jpg"
    size=$(file "$src" | grep -o '[0-9]\{3,5\}x[0-9]\{3,5\}' | tail -1)
    for mode in colour invert; do
        flag=""; [ "$mode" = invert ] && flag="--invert"
        out="$OUT_DIR/results/${name}_after_${mode}.jpg"
        ilog="$OUT_DIR/logs/${name}_${mode}.log"
        t0=$(date +%s.%N)
        "./$EXE" --input="$src" --output="$out" $flag > "$ilog" 2>&1
        rc=$?
        t1=$(date +%s.%N)
        secs=$(awk "BEGIN{printf \"%.2f\", $t1 - $t0}")
        hue=$(sed -n 's/.*Most common hue: \([0-9]*\) degrees.*/\1/p' "$ilog")
        kept=$(sed -n 's/.*kept \([0-9.e+-]*\)% .*/\1/p' "$ilog")
        status=OK; [ $rc -ne 0 ] && { status=FAIL; failed=$((failed + 1)); }
        echo "$name,$mode,${size%x*},${size#*x},$hue,$kept,$secs,$status" >> "$CSV"
        printf '[%3d/%d] %-14s %-6s %s  hue=%s deg  kept=%s%%  %ss  %s\n' \
            "$n" "$COUNT" "$name" "$mode" "$size" "$hue" "$kept" "$secs" "$status" | tee -a "$LOG"
    done
done
end_all=$(date +%s)

{
    echo
    echo "processed $n images ($((n * 2)) GPU runs), $failed failed, total $((end_all - start_all)) s"
} | tee -a "$LOG"
[ $failed -eq 0 ]
