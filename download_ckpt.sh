mkdir -p checkpoints

wget https://github.com/m-baralt/smile-to-bert/releases/download/v3.0/atomlevel_ckpt.zip -O checkpoints/atomlevel_ckpt.zip
wget https://github.com/m-baralt/smile-to-bert/releases/download/v3.0/finetuned_ckp.pth -O checkpoints/finetuned_ckp.pth
wget https://github.com/m-baralt/smile-to-bert/releases/download/v3.0/spe_ckpt.zip -O checkpoints/spe_ckpt.zip
wget https://github.com/m-baralt/smile-to-bert/releases/download/v3.0/spe_large_ckpt.zip -O checkpoints/spe_large_ckpt.zip
wget https://github.com/m-baralt/smile-to-bert/releases/download/v3.0/transformer_ckpt.ckpt -O checkpoints/transformer_ckpt.ckpt

unzip checkpoints/atomlevel_ckpt.zip -d checkpoints/
unzip checkpoints/spe_ckpt.zip -d checkpoints/
unzip checkpoints/spe_large_ckpt.zip -d checkpoints/

rm checkpoints/atomlevel_ckpt.zip
rm checkpoints/spe_ckpt.zip
rm checkpoints/spe_large_ckpt.zip
