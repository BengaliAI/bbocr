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


## Citation

You can cite bbOCR as:

```
@misc{zulkarnain2023bbocr,
      title={bbOCR: An Open-source Multi-domain OCR Pipeline for Bengali Documents}, 
      author={Imam Mohammad Zulkarnain and Shayekh Bin Islam and Md. Zami Al Zunaed Farabe and Md. Mehedi Hasan Shawon and Jawaril Munshad Abedin and Beig Rajibul Hasan and Marsia Haque and Istiak Shihab and Syed Mobassir and MD. Nazmuddoha Ansary and Asif Sushmit and Farig Sadeque},
      year={2023},
      eprint={2308.10647},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



