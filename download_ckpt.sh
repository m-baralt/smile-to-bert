mkdir -p ckpts

wget https://github.com/m-baralt/smile-to-bert/releases/download/v2.0/checkpoint_19.zip -O ckpts/checkpoint_19.zip

unzip ckpts/checkpoint_19.zip -d ckpts/

rm ckpts/checkpoint_19.zip

