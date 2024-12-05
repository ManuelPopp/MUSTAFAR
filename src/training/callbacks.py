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
# Imports

#-----------------------------------------------------------------------------|
# Classes
class Callback:
    """
    Base class for all callbacks.

    Methods
    -------
    __call__(model, val_loss, **kwargs)
        Primary interface for executing the callback during training.

    """
    def __call__(self, model, val_loss, **kwargs):
        pass

class EarlyStopping(Callback):
    """
    Early stopping callback to terminate training when validation loss
    does not improve.

    Parameters
    ----------
    patience : int, optional
        Number of epochs to wait without improvement before stopping
        training. Defaults to 10.

    Attributes
    ----------
    patience : int
        Number of epochs to wait before stopping training.
    best_loss : float
        The lowest validation loss encountered during training.
    best_model : dict or None
        The state dictionary of the best-performing model.
    counter : int
        Number of epochs since the last improvement in validation loss.

    Methods
    -------
    __call__(model, val_loss, **kwargs)
        Check the validation loss and determine whether to stop training.
    reset()
        Reset the state of the callback to its initial state.

    """
    def __init__(self, patience = 10):
        """
        Initialize the EarlyStopping callback.

        Parameters
        ----------
        patience : int, optional
            Number of epochs to wait without improvement before stopping
            training. Defaults to 10.

        """
        self.patience = patience
        self.reset()
    
    def __call__(self, model, val_loss, **kwargs):
        """
        Check the validation loss and decide whether to stop training.

        If validation loss improves, the state of the best-performing
        model is saved. If validation loss does not improve for
        `patience` consecutive epochs, training is stopped.

        Parameters
        ----------
        model : torch.nn.Module
            The model being trained.
        val_loss : float
            The validation loss for the current epoch.
        **kwargs : dict, optional
            Additional arguments for callback logic, such as
            "store_copy" for storing training history.

        Returns
        -------
        bool
            True if training should stop, False otherwise.

        """
        store_copy = kwargs.get("store_copy", None)
        
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model = {
                k: v.clone() for k, v in model.state_dict().items()
                }
            self.store_copy = store_copy
            self.counter = 0
        
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            print(f"Patience of {self.patience} exceeded. Training stopped.")
            return True
        
        return False
    
    def reset(self):
        """
        Reset the state of the callback.

        This method reinitialises the best loss, best model, and counter
        to allow the callback to be reused for a new training session.

        """
        self.best_loss = float("inf")
        self.best_model = None
        self.counter = 0
