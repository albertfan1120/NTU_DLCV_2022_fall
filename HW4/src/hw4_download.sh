mkdir -p p1/logs/nerf_synthetic/dvgo_hotdog/
wget -O p1/logs/nerf_synthetic/dvgo_hotdog/coarse_last.tar 'https://www.dropbox.com/s/g72sdjq2wxehnt3/coarse_last.tar?dl=1'
wget -O p1/logs/nerf_synthetic/dvgo_hotdog/fine_last.tar 'https://www.dropbox.com/s/ybayqf3a0xohpjo/fine_last.tar?dl=1'

mkdir p2/save_model/
wget -O p2/save_model/renet50_finetune.pth 'https://www.dropbox.com/s/15kzccu01lp5i15/renet50_finetune.pth?dl=1'