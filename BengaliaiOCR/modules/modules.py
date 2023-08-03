#-*- coding: utf-8 -*-
from __future__ import print_function
#-------------------------
# imports
#-------------------------
from abc import ABCMeta, abstractmethod

class Recognizer(metaclass=ABCMeta):
    """Recognizer base class
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def infer(self):
        pass

class Detector(metaclass=ABCMeta):
    """Recognizer base class
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def get_word_boxes(self):
        pass

    @abstractmethod
    def get_line_boxes(self):
        pass
    
    @abstractmethod
    def get_crops(self):
        pass