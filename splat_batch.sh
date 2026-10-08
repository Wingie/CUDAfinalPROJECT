#!/bin/bash
# Batch-runs splat (photo -> 3D Gaussians) over N random photos from every
# sub-folder of a photo library, for checking the depth pipeline on a wide variety of images.
#   data/ply/<folder>__<photo>.ply   one Gaussian PLY per photo (Space cycles through them in the viewer)
#   <log_dir>/summary.csv            folder, image, size, depth-model ms, Gaussians, edge points, seconds, status
#   <log_dir>/<folder>__<photo>.log  splat's full output per photo
#
# Run in WSL/Linux: ./splat_batch.sh [models_dir] [per_folder] [log_dir]
#   defaults: /mnt/d/images/models  10  data/splat_logs
# Extra settings come from the environment: MODEL, DEPTH_RANGE, EDGE, MAX_POINTS

IN_DIR=${1:-/mnt/d/images/models}
PER_FOLDER=${2:-10}
LOG_DIR=${3:-data/splat_logs}
MODEL=${MODEL:-depth_anything_v2_small.onnx}
DEPTH_RANGE=${DEPTH_RANGE:-10}
EDGE=${EDGE:-0.1}
MAX_POINTS=${MAX_POINTS:-500000}
OUT_DIR=data/ply

make -s bin/splat "$MODEL" || exit 1
mkdir -p "$OUT_DIR" "$LOG_DIR"
CSV="$LOG_DIR/summary.csv"
LOG="$LOG_DIR/run.log"
echo "folder,image,width,height,warm_ms,gaussians,edge_points,seconds,status" > "$CSV"

{
    echo "=== splat batch run ==="
    echo "date:   $(date)"
    echo "input:  $IN_DIR ($PER_FOLDER random photos per folder)"
    echo "model:  $MODEL  depth_range=$DEPTH_RANGE  edge=$EDGE  max_points=$MAX_POINTS"
    /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/gpu:    /'
    echo
} | tee "$LOG"

start_all=$(date +%s)
n=0; failed=0
while IFS= read -r -d '' dir; do
    folder=$(basename "$dir")
    tag=$(echo "$folder" | tr ' ' '_')
    while IFS= read -r -d '' src; do
        n=$((n + 1))
        name=$(basename "${src%.*}")
        out="$OUT_DIR/${tag}__${name}.ply"
        ilog="$LOG_DIR/${tag}__${name}.log"
        t0=$(date +%s.%N)
        ./bin/splat "$src" "$out" "$MODEL" "$MAX_POINTS" "$DEPTH_RANGE" "$EDGE" > "$ilog" 2>&1
        rc=$?
        secs=$(awk "BEGIN{printf \"%.1f\", $(date +%s.%N) - $t0}")
        size=$(sed -n 's/^Image \([0-9]*\)x\([0-9]*\),.*/\1,\2/p' "$ilog")
        warm=$(sed -n 's/^Depth inference (warm): \([0-9.]*\) ms/\1/p' "$ilog")
        kept=$(sed -n 's/.*kept \([0-9]*\) of \([0-9]*\) points (\([0-9]*\) on.*/\1,\3/p' "$ilog")
        status=OK; [ $rc -ne 0 ] && { status=FAIL; failed=$((failed + 1)); }
        echo "\"$folder\",$name,${size:-,},$warm,${kept:-,},$secs,$status" >> "$CSV"
        printf '[%3d] %-22s %-14s %-10s %6s ms  kept %-14s %5ss  %s\n' \
            "$n" "$folder" "$name" "${size/,/x}" "$warm" "${kept/,/ edges }" "$secs" "$status" | tee -a "$LOG"
    done < <(find "$dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print0 \
             | shuf -z -n "$PER_FOLDER")
done < <(find "$IN_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

{
    echo
    echo "processed $n photos, $failed failed, total $(( $(date +%s) - start_all )) s"
} | tee -a "$LOG"
[ $failed -eq 0 ]
