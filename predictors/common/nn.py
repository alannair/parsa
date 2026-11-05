# Copyright (C) 2026 M.R. Siavash Katebzadeh <mr@katebzadeh.xyz>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import copy
import os
import tempfile

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import get as get_optimizer

from predictors.common.features import FEATURE_COLS, ensure_array


def validate_training_frame(df, target_col, feature_cols=None):
    feature_cols = FEATURE_COLS if feature_cols is None else feature_cols
    if df.empty:
        raise ValueError("DataFrame is empty")
    missing = set(feature_cols + [target_col]) - set(df.columns)
    if missing:
        raise KeyError("DataFrame missing required columns: %s" % missing)


def train_benchmark_nn(
    df,
    target_col,
    *,
    layer_sizes,
    loss_function="mean_squared_error",
    optimizer="adam",
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    test_size=0.2,
    random_state=42,
    learning_rate=None,
    verbose=0,
    npu_ids=None,
    feature_cols=None,
):
    feature_cols = FEATURE_COLS if feature_cols is None else feature_cols
    validate_training_frame(df, target_col, feature_cols=feature_cols)
    if not isinstance(layer_sizes, (list, tuple)) or not layer_sizes or not all(
        isinstance(x, (int, np.integer)) and int(x) > 0 for x in layer_sizes
    ):
        raise ValueError("layer_sizes must be a non-empty list of positive integers")
    if not (0.0 < float(validation_split) < 1.0):
        raise ValueError("validation_split must be in (0, 1), got %r" % validation_split)

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    if npu_ids is not None:
        return _train_benchmark_nn_torch(
            X_train_s,
            y_train,
            X_test,
            X_test_s,
            y_test,
            feature_cols=feature_cols,
            target_col=target_col,
            layer_sizes=layer_sizes,
            loss_function=loss_function,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            random_state=random_state,
            learning_rate=learning_rate,
            verbose=verbose,
            npu_ids=npu_ids,
            scaler=scaler,
        )

    return _train_benchmark_nn_keras(
        X_train_s,
        y_train,
        X_test,
        X_test_s,
        y_test,
        feature_cols=feature_cols,
        target_col=target_col,
        layer_sizes=layer_sizes,
        loss_function=loss_function,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        learning_rate=learning_rate,
        verbose=verbose,
        scaler=scaler,
    )


def _build_predict_fn(model, feature_cols, target_col, *, scaler=None, use_scaler=False):
    def predict(samples, fitted_model=model, feature_cols=feature_cols, target_col=target_col):
        if isinstance(samples, list) and samples and isinstance(samples[0], dict):
            arr = np.array([[sample[k] for k in feature_cols] for sample in samples], dtype=np.float32)
        else:
            arr = ensure_array(samples, feature_cols)
        if use_scaler:
            arr = scaler.transform(arr).astype(np.float32)
        preds = np.asarray(fitted_model.predict(arr, verbose=0), dtype=np.float64).ravel()
        return [{target_col: float(preds[index])} for index in range(len(preds))]

    return predict


def _train_result(
    *,
    model,
    X_test,
    y_test,
    y_pred,
    feature_cols,
    target_col,
    predict,
    fit_history,
    best_epoch,
    best_val_loss,
    scaler=None,
    device=None,
    npu_ids=None,
    torch_model=None,
    layer_sizes=None,
):
    y_true = np.asarray(y_test, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_true,
        "y_true": y_true,
        "y_pred": y_pred,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "predict": predict,
        "fit_history": fit_history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "scaler": scaler,
        "device": device,
        "npu_ids": npu_ids,
        "torch_model": torch_model,
        "model_type": "nn",
        "backend": "torch" if torch_model is not None else "keras",
        "layer_sizes": list(layer_sizes) if layer_sizes is not None else None,
    }


def _train_benchmark_nn_torch(
    X_train_s,
    y_train,
    X_test,
    X_test_s,
    y_test,
    *,
    feature_cols,
    target_col,
    layer_sizes,
    loss_function,
    optimizer,
    epochs,
    batch_size,
    validation_split,
    random_state,
    learning_rate,
    verbose,
    npu_ids,
    scaler,
):
    import torch
    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.utils.data import TensorDataset

    if len(npu_ids) != 1:
        raise ValueError(
            "Ascend NPU training only supports a single device; pass npu_ids with exactly one id, e.g. [0]."
        )
    try:
        import torch_npu
    except ImportError as exc:
        raise ImportError(
            "Ascend NPU training requires torch_npu (install the CANN-matched wheel for your PyTorch)."
        ) from exc
    if not torch.npu.is_available():
        raise RuntimeError(
            "npu_ids was set but no Ascend NPU is visible (torch.npu.is_available() is False)."
        )

    npu_idx = int(npu_ids[0])
    torch.npu.set_device(npu_idx)
    device = torch.device("npu:%d" % npu_idx)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_s,
        y_train.astype(np.float32),
        test_size=validation_split,
        random_state=random_state,
    )
    in_dim = int(X_tr.shape[1])

    if loss_function in ("mean_squared_error", "mse"):
        criterion = torch.nn.MSELoss()
    elif loss_function in ("mean_absolute_error", "mae"):
        criterion = torch.nn.L1Loss()
    else:
        raise ValueError(
            "For npu_ids path, loss_function must be one of "
            "{'mean_squared_error','mse','mean_absolute_error','mae'}"
        )

    opt_name = optimizer if isinstance(optimizer, str) else None
    if opt_name not in ("adam", "rmsprop", None):
        raise ValueError("For npu_ids path, optimizer must be 'adam' or 'rmsprop'")

    torch.manual_seed(int(random_state))

    layers = []
    prev = in_dim
    for hidden in layer_sizes:
        hidden = int(hidden)
        layers.append(torch.nn.Linear(prev, hidden))
        layers.append(torch.nn.ReLU())
        prev = hidden
    layers.append(torch.nn.Linear(prev, 1))

    torch_model = torch.nn.Sequential(*layers).to(device)

    lr = 1e-3 if learning_rate is None else float(learning_rate)
    if opt_name == "rmsprop":
        opt = torch.optim.RMSprop(torch_model.parameters(), lr=lr)
    else:
        opt = torch.optim.Adam(torch_model.parameters(), lr=lr)

    train_loader = TorchDataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32))),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )
    val_loader = TorchDataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val.astype(np.float32))),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
    )

    train_losses = []
    val_losses = []
    best_state = None
    best_val = float("inf")

    for epoch in range(int(epochs)):
        torch_model.train()
        tr_total = 0.0
        tr_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = torch_model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            tr_total += float(loss.item()) * xb.size(0)
            tr_n += int(xb.size(0))
        train_losses.append(tr_total / max(tr_n, 1))

        torch_model.eval()
        va_total = 0.0
        va_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = torch_model(xb).squeeze(-1)
                loss = criterion(pred, yb)
                va_total += float(loss.item()) * xb.size(0)
                va_n += int(xb.size(0))
        va = va_total / max(va_n, 1)
        val_losses.append(va)

        if va < best_val:
            best_val = va
            best_state = copy.deepcopy(torch_model.state_dict())

        if verbose:
            print(
                "epoch %d/%d: loss=%.6f val_loss=%.6f"
                % (epoch + 1, int(epochs), train_losses[-1], val_losses[-1])
            )

    if best_state is not None:
        torch_model.load_state_dict(best_state)

    best_idx = int(np.argmin(val_losses))
    best_epoch = best_idx + 1
    best_val_loss = float(val_losses[best_idx])
    fit_history = {"loss": train_losses, "val_loss": val_losses}

    class _TorchModelWrapper:
        def __init__(self, module, dev, fitted_scaler):
            self._module = module
            self._dev = dev
            self._scaler = fitted_scaler

        def predict(self, X_in, verbose=0):
            arr = np.asarray(X_in, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            arr_s = self._scaler.transform(arr).astype(np.float32)
            x = torch.from_numpy(arr_s).to(self._dev)
            self._module.eval()
            with torch.no_grad():
                out = self._module(x).squeeze(-1).detach().cpu().numpy()
            return out

    wrapped_model = _TorchModelWrapper(torch_model, device, scaler)
    y_pred = np.asarray(wrapped_model.predict(X_test, verbose=0), dtype=np.float64)
    predict = _build_predict_fn(wrapped_model, feature_cols, target_col, scaler=scaler, use_scaler=True)

    return _train_result(
        model=wrapped_model,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        feature_cols=feature_cols,
        target_col=target_col,
        predict=predict,
        fit_history=fit_history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        scaler=scaler,
        device=device,
        npu_ids=list(npu_ids),
        torch_model=torch_model,
        layer_sizes=layer_sizes,
    )


def _train_benchmark_nn_keras(
    X_train_s,
    y_train,
    X_test,
    X_test_s,
    y_test,
    *,
    feature_cols,
    target_col,
    layer_sizes,
    loss_function,
    optimizer,
    epochs,
    batch_size,
    validation_split,
    learning_rate,
    verbose,
    scaler,
):
    model = Sequential()
    model.add(Dense(int(layer_sizes[0]), input_dim=X_train_s.shape[1], activation="relu"))
    for hidden in layer_sizes[1:]:
        model.add(Dense(int(hidden), activation="relu"))
    model.add(Dense(1, activation="linear"))

    opt = optimizer
    if isinstance(opt, str):
        opt = get_optimizer(opt)
    if learning_rate is not None and hasattr(opt, "learning_rate"):
        try:
            opt.learning_rate.assign(float(learning_rate))
        except Exception:
            opt.learning_rate = float(learning_rate)

    model.compile(optimizer=opt, loss=loss_function)

    ckpt_fd, ckpt_path = tempfile.mkstemp(suffix=".weights.h5")
    os.close(ckpt_fd)
    callbacks = [
        ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        )
    ]
    try:
        history = model.fit(
            X_train_s,
            y_train,
            epochs=int(epochs),
            batch_size=int(batch_size),
            validation_split=float(validation_split),
            callbacks=callbacks,
            verbose=verbose,
        )
        model.load_weights(ckpt_path)
    finally:
        try:
            os.remove(ckpt_path)
        except OSError:
            pass

    val_losses = history.history.get("val_loss", [])
    best_epoch = int(np.argmin(val_losses)) + 1 if val_losses else None
    best_val_loss = float(np.min(val_losses)) if val_losses else None
    y_pred = np.asarray(model.predict(X_test_s, verbose=verbose), dtype=np.float64).ravel()
    predict = _build_predict_fn(model, feature_cols, target_col, scaler=scaler, use_scaler=True)

    return _train_result(
        model=model,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        feature_cols=feature_cols,
        target_col=target_col,
        predict=predict,
        fit_history=dict(history.history),
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        scaler=scaler,
        device=None,
        npu_ids=None,
        torch_model=None,
        layer_sizes=layer_sizes,
    )
