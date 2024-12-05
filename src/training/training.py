#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Manuel"
__date__ = "Wed Dec 04 11:34:15 2024"
__credits__ = ["Manuel R. Popp"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Manuel R. Popp"
__email__ = "requests@cdpopp.de"
__status__ = "Development"

#-----------------------------------------------------------------------------|
# Imports
import torch
from tqdm import tqdm
from warnings import warn

#-----------------------------------------------------------------------------|
# Settings
default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-----------------------------------------------------------------------------|
# Classes
def train(
    model, train_loader, validation_loader, epochs, loss_function,
    callbacks = None, optimizer = None, device = default_device,
    filename = None, output = "pandas"
    ):
    """
    Run model training for a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be trained.
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training set.
    validation_loader : torch.utils.data.DataLoader
        DataLoader for the validation set.
    epochs : int
        Number of epochs to train the model.
    loss_function : torch.nn.modules.loss._Loss
        Loss function to optimise.
    callbacks : list[Callback], optional
        List of callbacks to use during training. Defaults to None.
    optimizer : torch.optim.Optimizer, optional Optimizer for training.
        If set to None, the Adam optimiser is used. Defaults to None.
    device : torch.device, optional
        Device to use for training. Defaults to default_device.
    filename : str, optional
        Filename to save the model. Defaults to None.
    output : str, optional
        Format of training history ("pandas" or "dict"). Defaults to "pandas".

    Returns
    -------
    pd.DataFrame or dict
        Training history as a pandas DataFrame or dictionary.

    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())
    
    history = {
        "training_loss": [],
        "validation_loss": [],
        "training_accuracy": [],
        "validation_accuracy": []
        }
    
    model.to(device)
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        progress_bar = tqdm(
            train_loader, desc = f"Epoch {epoch + 1}/{epochs} [Train]",
            leave = False
            )
        
        for x_batch, y_batch in progress_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            y_pred = model(x_batch)
            loss = loss_function(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            correct_train += (y_pred.argmax(1) == y_batch).sum().item()
            total_train += y_batch.size(0)
            
            progress_bar.set_postfix(
                loss = running_loss / total_train,
                accuracy = correct_train / total_train
                )
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        validation_bar = tqdm(
            validation_loader,
            desc = f"Epoch {epoch + 1}/{epochs} [Validation]", leave = False
            )
        
        # Calculate validation loss and accuracy
        with torch.no_grad():
            for x_val, y_val in validation_bar:
                x_val, y_val = x_val.to(device), y_val.to(device)
                y_pred = model(x_val)
                loss = loss_function(y_pred, y_val)
                val_loss += loss.item()
                _, predicted = torch.max(y_pred, 1)
                correct += (predicted == y_val).sum().item()
                total += y_val.size(0)
                validation_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
        
        val_loss /= len(validation_loader)
        val_accuracy = correct / total

        history["training_loss"].append(train_loss)
        history["training_accuracy"].append(train_accuracy)
        history["validation_loss"].append(val_loss)
        history["validation_accuracy"].append(val_accuracy)
        
        if callbacks:
            # Ensure callbacks are iterable
            if not isinstance(callbacks, list):
                callbacks = [callbacks]
            
            # Check callbacks
            for callback in callbacks:
                stop_training = callback(
                    model, val_loss, store_copy = history
                    )
            
            if stop_training:
                for callback in callbacks:
                    # Restore best model state
                    if hasattr(callback, "best_model"):
                        print("Restoring best model.")
                        model.load_state_dict(callback.best_model)
                    
                    # Restore training history until break
                    if hasattr(callback, "store_copy"):
                        history = callback.store_copy
                break
    
    # Convert history to pandas DataFrame
    if output == "pandas":
        try:
            from pandas import DataFrame
            history = DataFrame(history)
        except ImportError:
            print(
                "Pandas is not installed. " +
                "Training history is returned as a dictionary."
                )
    
    # Save model
    if filename:
        try:
            torch.save(model.state_dict(), filename)
        except Exception as e:
            warn(f"Failed to save model: {e}. Model is returned instead.")
    
    return model, history
