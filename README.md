# MED-Agents
AVIS for cancer data


_1. TO DOWNLOAD IMAGES_

cd Slides/

 ./gdc-client download -m gdc_manifest_20231226_105105.txt

_2. CREATE ENV_

create python evn

pip install pandas

pip install timm==0.5.4

_3. DOWNLOAD WEIGHTS and RUN MODEL_

cd TransPath/

download the pretrained model here ctranspath.pth

python get_features_CTransPath.py
