mkdir -p ckpts

wget https://github.com/m-baralt/smile-to-bert/releases/download/v1.0/all4M_data.zip -O data/all4M_data.zip

unzip data/all4M_data.zip -d data/

rm data/all4M_data.zip

