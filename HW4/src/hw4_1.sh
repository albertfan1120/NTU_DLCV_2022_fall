cd p1
python3 inference.py --json_root $1 --save_path $2 --config configs/nerf/hotdog.py --render_test
cd ..