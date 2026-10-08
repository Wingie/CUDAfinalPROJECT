rm -f data/ply/*.ply
for f in IMG_1515 IMG_1516 IMG_1517 IMG_1520; do
  make -s run-splat INPUT=/mnt/d/images/models/charlize/$f.jpg 2>&1 | grep -E "^Image"
done
make -s run-splat INPUT=data/Lena.png 2>&1 | grep -E "^Image"
make -s run-splat INPUT=/mnt/d/images/models/charlize/IMG_1515.jpg MODEL=midas.onnx PLY=data/ply/IMG_1515_midas.ply 2>&1 | grep -E "^Image"
bash splat_batch.sh
