# Modules 

## Apsisnet

* getting the weight files : in current directory

```python
import os 
from BengaliaiOCR.utils import download
bnocr_onnx="bnocr.onnx",
bnocr_gid="1YwpcDJmeO5mXlPDj1K0hkUobpwGaq3YA"
if not os.path.exists(bnocr_onnx):
    download(bnocr_gid,bnocr_onnx)
```
* import

```python
from BengaliaiOCR import ApsisNet
```

* initialization 

```python
recognizer=ApsisNet(model_weights=bnocr_onnx)
```

* call 

```python
texts=recognizer(crops,batch_size=32,normalize_unicode=True)
```
