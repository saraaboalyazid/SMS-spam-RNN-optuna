import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm

from ..core.torch_utils import moveTo


def run_epoch(
    model,
    optimizer,
    data_loader,
    loss_func,
    device,
    results,
    score_funcs,
    prefix="",
    desc=None,
):
    """
    model -- the PyTorch model / "Module" to run for one epoch
    optimizer -- the object that will update the weights of the network
    data_loader -- DataLoader object that returns tuples of (input, label) pairs.
    loss_func -- the loss function that takes in two arguments, the model outputs and the labels, and returns a score
    device -- the compute lodation to perform training
    score_funcs -- a dictionary of scoring functions to use to evalue the performance of the model
    prefix -- a string to pre-fix to any scores placed into the _results_ dictionary.
    desc -- a description to use for the progress bar.
    """
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()
    for inputs, labels, lengths in tqdm(data_loader, desc=desc, leave=False):
        # Move the batch to the device we are using.
        inputs = moveTo(inputs, device)
        labels = moveTo(labels, device)

        y_hat = model(inputs)  # this just computed f_Θ(x(i))
        # Compute loss.
        loss = loss_func(y_hat, labels)

        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Now we are just grabbing some information we would like to have
        running_loss.append(loss.item())

        if len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
            # moving labels & predictions back to CPU for computing / storing predictions
            labels = labels.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()
            # add to predictions so far
            y_true.extend(labels.tolist())
            y_pred.extend(y_hat.tolist())
    # end training epoch
    end = time.time()

    y_pred = np.asarray(y_pred)
    if (
        len(y_pred.shape) == 2 and y_pred.shape[1] > 1
    ):  # We have a classification problem, convert to labels
        y_pred = np.argmax(y_pred, axis=1)
    # Else, we assume we are working on a regression problem

    results[prefix + " loss"].append(np.mean(running_loss))
    for name, score_func in score_funcs.items():
        try:
            results[prefix + " " + name].append(score_func(y_true, y_pred))
        except:
            results[prefix + " " + name].append(float("NaN"))
    return end - start  # time spent on epoch


def train_simple_network(
    model,
    loss_func,
    train_loader,
    test_loader=None,
    score_funcs=None,
    epochs=50,
    device="cpu",
    checkpoint_file=None,
    lr=0.001,
):
    """Train simple neural networks

    Keyword arguments:
    model -- the PyTorch model / "Module" to train
    loss_func -- the loss function that takes in batch in two arguments, the model outputs and the labels, and returns a score
    train_loader -- PyTorch DataLoader object that returns tuples of (input, label) pairs.
    test_loader -- Optional PyTorch DataLoader to evaluate on after every epoch
    score_funcs -- A dictionary of scoring functions to use to evalue the performance of the model
    epochs -- the number of training epochs to perform
    device -- the compute lodation to perform training

    """
    to_track = ["epoch", "total time", "train loss"]
    if test_loader is not None:
        to_track.append("test loss")
    for eval_score in score_funcs:
        to_track.append("train " + eval_score)
        if test_loader is not None:
            to_track.append("test " + eval_score)

    total_train_time = 0  # How long have we spent in the training loop?
    results = {}
    # Initialize every item with an empty list
    for item in to_track:
        results[item] = []

    # SGD is Stochastic Gradient Decent.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # Place the model on the correct compute resource (CPU or GPU)
    model.to(device)
    for epoch in tqdm(range(epochs), desc="Epoch"):
        model = model.train()  # Put our model in training mode

        total_train_time += run_epoch(
            model,
            optimizer,
            train_loader,
            loss_func,
            device,
            results,
            score_funcs,
            prefix="train",
            desc="Training",
        )

        results["total time"].append(total_train_time)
        results["epoch"].append(epoch)

        if test_loader is not None:
            model = model.eval()
            with torch.no_grad():
                run_epoch(
                    model,
                    optimizer,
                    test_loader,
                    loss_func,
                    device,
                    results,
                    score_funcs,
                    prefix="test",
                    desc="Testing",
                )

    if checkpoint_file is not None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "results": results,
            },
            checkpoint_file,
        )

    return pd.DataFrame.from_dict(results)


def train_network(
    model,
    loss_func,
    train_loader,
    val_loader=None,
    test_loader=None,
    score_funcs=None,
    epochs=50,
    device="cpu",
    checkpoint_file=None,
    lr_schedule=None,
    optimizer=None,
    disable_tqdm=False,
):
    """Train simple neural networks

    Keyword arguments:
    model -- the PyTorch model / "Module" to train
    loss_func -- the loss function that takes in batch in two arguments, the model outputs and the labels, and returns a score
    train_loader -- PyTorch DataLoader object that returns tuples of (input, label) pairs.
    val_loader -- Optional PyTorch DataLoader to evaluate on after every epoch
    test_loader -- Optional PyTorch DataLoader to evaluate on after every epoch
    score_funcs -- A dictionary of scoring functions to use to evalue the performance of the model
    epochs -- the number of training epochs to perform
    device -- the compute lodation to perform training
    lr_schedule -- the learning rate schedule used to alter \eta as the model trains. If this is not None than the user must also provide the optimizer to use.
    optimizer -- the method used to alter the gradients for learning.

    """
    if score_funcs == None:
        score_funcs = {}  # Empty set

    to_track = ["epoch", "total time", "train loss"]
    if val_loader is not None:
        to_track.append("val loss")
    if test_loader is not None:
        to_track.append("test loss")
    for eval_score in score_funcs:
        to_track.append("train " + eval_score)
        if val_loader is not None:
            to_track.append("val " + eval_score)
        if test_loader is not None:
            to_track.append("test " + eval_score)

    total_train_time = 0  # How long have we spent in the training loop?
    results = {}
    # Initialize every item with an empty list
    for item in to_track:
        results[item] = []

    if optimizer == None:
        # The AdamW optimizer is a good default optimizer
        optimizer = torch.optim.AdamW(model.parameters())

    # Place the model on the correct compute resource (CPU or GPU)
    model.to(device)
    for epoch in tqdm(range(epochs), desc="Epoch", disable=disable_tqdm):
        model = model.train()  # Put our model in training mode

        total_train_time += run_epoch(
            model,
            optimizer,
            train_loader,
            loss_func,
            device,
            results,
            score_funcs,
            prefix="train",
            desc="Training",
        )

        results["epoch"].append(epoch)
        results["total time"].append(total_train_time)

        if val_loader is not None:
            model = (
                model.eval()
            )  # Set the model to "evaluation" mode, b/c we don't want to make any updates!
            with torch.no_grad():
                run_epoch(
                    model,
                    optimizer,
                    val_loader,
                    loss_func,
                    device,
                    results,
                    score_funcs,
                    prefix="val",
                    desc="Validating",
                )

        # In PyTorch, the convention is to update the learning rate after every epoch
        if lr_schedule is not None:
            if isinstance(lr_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_schedule.step(results["val loss"][-1])
            else:
                lr_schedule.step()

        if test_loader is not None:
            model = (
                model.eval()
            )  # Set the model to "evaluation" mode, b/c we don't want to make any updates!
            with torch.no_grad():
                run_epoch(
                    model,
                    optimizer,
                    test_loader,
                    loss_func,
                    device,
                    results,
                    score_funcs,
                    prefix="test",
                    desc="Testing",
                )

        if checkpoint_file is not None:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "results": results,
                },
                checkpoint_file,
            )

    return pd.DataFrame.from_dict(results)


class Flatten(nn.Module):
    """
    A custom PyTorch layer that flattens the input tensor.

    Typically used before a fully connected (Linear) layer in CNNs,
    to convert multi-dimensional feature maps into a 2D tensor:
    [batch_size, flattened_features].
    """

    def forward(self, input):
        # input: Tensor of shape [batch_size, channels, height, width]
        # view(): reshapes the tensor while keeping the batch dimension the same
        # input.size(0): keeps the batch size
        # -1: flattens all remaining dimensions automatically
        return input.view(input.size(0), -1)


class View(nn.Module):
    """
    A flexible PyTorch layer that reshapes (views) the input tensor
    to a specified shape.

    Unlike Flatten, it allows you to reshape into *any* target shape
    (as long as the total number of elements stays the same).
    """

    def __init__(self, *shape):
        """
        Parameters
        ----------
        *shape : tuple
            The desired output shape (e.g., (-1, 128), (batch, 16, 8, 8), etc.)
        """
        super(View, self).__init__()
        self.shape = shape  # store the target shape

    def forward(self, input):
        # input: Tensor of any shape
        # view(*self.shape): reshapes the tensor according to self.shape
        # The * operator unpacks the tuple into separate arguments
        return input.view(*self.shape)
