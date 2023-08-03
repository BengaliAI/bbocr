# Modules 

## Apsisnet

```python
from BengaliaiOCR import ApsisNet
recognizer=ApsisNet()
texts=recognizer.infer(crops,batch_size=32,normalize_unicode=True)
```
## Detector

```python
# initialization
from BengaliaiOCR import PaddleDBNet
detector=PaddleDBNet(use_gpu=False)
# getting word boxes
word_boxes=detector.get_word_boxes(img)
# getting line boxes
line_boxes=detector.get_line_boxes(img)
# getting crop with either of the results
crops=detector.get_crops(img,word_boxes)
```