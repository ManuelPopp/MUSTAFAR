#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Manuel"
__date__ = "Wed Dec 04 11:20:41 2024"
__credits__ = ["Manuel R. Popp"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Manuel R. Popp"
__email__ = "requests@cdpopp.de"
__status__ = "Development"

#-----------------------------------------------------------------------------|
import os, sys, torch

class EarlyStopping:
    def __init__(self, patience = 10, max_no_improve = 10):
        self.patience = patience
        self.max_no_improve = max_no_improve
        self.best_loss = float("inf")
        self.best_model = None
        self.counter = 0
        self.no_improve_counter = 0
    
    def __call__(self, model, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model = {
                k: v.clone() for k, v in model.state_dict().items()
                }
            self.counter = 0
            self.no_improve_counter = 0
        else:
            self.counter += 1
            self.no_improve_counter += 1
        
        if self.counter >= self.patience:
            print(f"Patience of {self.patience} exceeded. Training stopped.")
            return True
        if self.no_improve_counter >= self.max_no_improve:
            print(
                f"No progress has been made for {self.max_no_improve} epochs." +
                " Training stopped."
                )
            return True
        
        return False