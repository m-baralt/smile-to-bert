mkdir -p data_spe

wget https://github.com/m-baralt/smile-to-bert/releases/download/v2.0/data_tensors.zip -O data_spe/data_tensors.zip
wget https://github.com/m-baralt/smile-to-bert/releases/download/v2.0/filt_props_tensor.zip -O data_spe/filt_props_tensor.zip

unzip data_spe/data_tensors.zip -d data_spe/
unzip data_spe/filt_props_tensor.zip -d data_spe/
