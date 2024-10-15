mkdir -p data_atomlevel

wget https://github.com/m-baralt/smile-to-bert/releases/download/v1.0/data_tensors.zip -O data_atomlevel/data_tensors.zip
wget https://github.com/m-baralt/smile-to-bert/releases/download/v1.0/filt_props_tensor.zip -O data_atomlevel/filt_props_tensor.zip

unzip data_atomlevel/data_tensors.zip -d data_atomlevel/
unzip data_atomlevel/filt_props_tensor.zip -d data_atomlevel/
