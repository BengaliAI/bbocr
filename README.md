<!-- # ocr

## TODO
- [ ] populate isolated standalone modules under **BengaliaiOCR/modules**
- [ ] create a ocr class in **BengaliaiOCR/ocr.py**
    - [ ] 2 public functions: ```eval() and infer()```
    - [ ] 2 modes for eval: ```pipeline and standalone``` -->

# bbOCR: An Open-source Multi-domain OCR Pipeline for Bengali Documents




## Installation

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install paddlepaddle-gpu==2.5.0 cudatoolkit=11.6 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge -y
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
pip install paddleocr ultralytics layoutparser bnunicodenormalizer onnxruntime-gpu
```






