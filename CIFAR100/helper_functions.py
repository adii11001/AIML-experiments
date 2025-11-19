from timeit import default_timer as timer
import requests
from pathlib import Path
from tqdm.auto import tqdm
import torch
from torch import nn

def train_time(start: float, end: float, device: torch.device):
  """ Function to compute the total train time
  Args: 
  start (float): training start time
  end (float): training end time 

  Returns: 
  The total train time 
  """
  total_time = end - start
  print(f"Took {total_time} seconds on device {device}")
  return total_time

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def train_step(model: nn.Module, 
               train_dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device):
  """ A single epoch of training
  Args: 
  model (nn.Module): The model 
  train_dataloader (torch.utils.data.DataLoader): The dataloader with training data
  loss_fn (nn.Module): loss function 
  optimizer (torch.optim.Optimizer): optimizer for updating parameters
  device (torch.device): device on which training is done
  accuracy_fn: function to compute accuracy of predictions
  """
  model.to(device)
  model.train()
  batch_train_acc, batch_train_loss = 0, 0
  for X, y in tqdm(train_dataloader, desc="Training in progress"):
    X, y = X.to(device), y.to(device)
    # 1. Forward pass
    train_pred = model(X)
    train_pred_label = train_pred.argmax(dim=1)

    # 2. Calculate Loss
    train_loss = loss_fn(train_pred, y)
    batch_train_loss += train_loss.item()
    batch_train_acc += accuracy_fn(y, train_pred_label)

    # 3. Zero grad
    optimizer.zero_grad()

    # 4. Backpropagation
    train_loss.backward()

    # 5. Optimizer step 
    optimizer.step()

  batch_train_acc /= len(train_dataloader)
  batch_train_loss /= len(train_dataloader)

  print(f"Train loss: {batch_train_loss:.4f} | Train acc: {batch_train_acc:.2f}%")

def test_step(model: nn.Module, 
              test_dataloader: torch.utils.data.DataLoader,
              loss_fn: nn.Module,
              accuracy_fn,
              device: torch.device):
  """ A single epoch of testing
  Args: 
  model (nn.Module): The model 
  test_dataloader (torch.utils.data.DataLoader): The dataloader with testing data
  loss_fn (nn.Module): loss function 
  device (torch.device): device on which testing is done
  accuracy_fn: function to compute accuracy of predictions
  """  
  model.eval()
  batch_test_loss, batch_test_acc = 0, 0
  with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc="Testing in progress"):
      X, y = X.to(device), y.to(device)
      # 1. Forward pass
      test_pred = model(X)
      test_pred_label = test_pred.argmax(dim=1)

      # 2. Calculate Loss
      test_loss = loss_fn(test_pred, y)
      batch_test_loss += test_loss.item()
      batch_test_acc += accuracy_fn(y, test_pred_label)

  batch_test_acc /= len(test_dataloader)
  batch_test_loss /= len(test_dataloader)
  print(f"Test loss: {batch_test_loss:.4f} | Test acc: {batch_test_acc:.2f}%")

def eval_model(model: nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               accuracy_fn,
               device: torch.device):
  """ Evaluates a model 
  Args: 
  model (nn.Module): The model 
  data_loader (torch.utils.data.DataLoader): The dataloader with testing data
  loss_fn (nn.Module): loss function 
  accuracy_fn: function to compute accuracy of predictions
  device (torch.device): device on which testing is done

  Returns: A dictionary containing model_name, model_acc, model_loss
  """
  model.eval()
  batch_test_loss, batch_test_acc = 0, 0
  with torch.inference_mode():
    for X, y in tqdm(data_loader):
      X, y = X.to(device), y.to(device)
      # 1. Forward pass
      test_pred = model(X)
      test_pred_label = test_pred.argmax(dim=1)

      # 2. Calculate Loss
      test_loss = loss_fn(test_pred, y)
      batch_test_loss += test_loss.item()
      batch_test_acc += accuracy_fn(y, test_pred_label)

  batch_test_acc /= len(data_loader)
  batch_test_loss /= len(data_loader)

  return {"model_name": model.__class__.__name__,
          "model_acc": batch_test_acc,
          "model_loss": batch_test_loss}

