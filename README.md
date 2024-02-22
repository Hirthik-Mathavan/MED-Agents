# MED-Tools

**1. Creating Environment**

* conda env create -n OOD --file transpath.yml
* Active env OOD

**2. DOWNLOADING IMAGES**

* change directory: cd Slides/
* run: ./gdc-client download -m gdc_manifest_20231226_105105.txt
* run: prep.py

**3. Downloading weights and running the model**

* cd TransPath/
* download the pretrained model here [ctranspath.pth](https://drive.google.com/file/d/1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX/view)
* python get_features_CTransPath.py
* python testing.py
