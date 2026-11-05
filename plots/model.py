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
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertModel, DataCollatorWithPadding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from predictors.common.metrics import as_1d_float as _as_1d_float
from predictors.common.sklearn import SKLEARN_GB_LOSS_MSE as _SKLEARN_GB_LOSS_MSE

_XGB_OBJECTIVE_MSE = "reg:squarederror"


def duplicate_singleton_generated_tokens(
    df,
    value_col="GeneratedTokens",
    n_duplicates=1,
    ignore_index=True,
):
    if value_col not in df.columns:
        raise KeyError(f"value_col {value_col!r} not in DataFrame columns")
    if df.empty:
        raise ValueError("DataFrame is empty")
    if n_duplicates < 0:
        raise ValueError(f"n_duplicates must be >= 0, got {n_duplicates}")

    counts = df[value_col].value_counts()
    singleton_vals = counts[counts == 1].index
    mask = df[value_col].isin(singleton_vals)
    singleton_rows = df.loc[mask]

    if n_duplicates == 0 or singleton_rows.empty:
        return {
            "df": df.copy(),
            "singleton_values": np.asarray(singleton_vals, dtype=object),
            "n_singleton_rows": int(singleton_rows.shape[0]),
            "n_rows_appended": 0,
        }

    extra = pd.concat([singleton_rows] * n_duplicates, ignore_index=ignore_index)
    out = pd.concat([df, extra], ignore_index=ignore_index)

    return {
        "df": out,
        "singleton_values": np.asarray(singleton_vals, dtype=object),
        "n_singleton_rows": int(singleton_rows.shape[0]),
        "n_rows_appended": int(singleton_rows.shape[0] * n_duplicates),
    }


def train_seq_pred_boost(df, test_size=0.2, random_state=42):
    required_cols = {"ContextTokens", "type", "GeneratedTokens"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame missing required columns: {missing}")
    if df.empty:
        raise ValueError("DataFrame is empty")

    feature_cols = ["ContextTokens", "type"]
    X = df[feature_cols].copy()
    y = df["GeneratedTokens"].to_numpy(dtype=np.float64)

    y_series = pd.Series(y)


    print("Class distribution:")
    print(y_series.value_counts())
    unique_classes, class_counts = np.unique(y, return_counts=True)


    one_sample_classes = unique_classes[class_counts == 1]


    num_one_sample_classes = len(one_sample_classes)

    print(f"Classes with only one sample: {one_sample_classes}")
    print(f"Number of classes with only one sample: {num_one_sample_classes}")

    preprocessor = ColumnTransformer(
        [
            ("context", StandardScaler(), ["ContextTokens"]),
            ("type", OneHotEncoder(handle_unknown="ignore"), ["type"]),
        ],
        remainder="drop",
    )
    model = Pipeline(
        [
            ("pre", preprocessor),
            ("reg", GradientBoostingRegressor(random_state=random_state, loss=_SKLEARN_GB_LOSS_MSE)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model.fit(X_train, y_train)

    y_pred = _as_1d_float(model.predict(X_test))

    def predict_generated_tokens(samples, fitted_model=model):
        if isinstance(samples, pd.DataFrame):
            X_in = samples[feature_cols]
        elif isinstance(samples, list) and samples and isinstance(samples[0], dict):
            X_in = pd.DataFrame(samples, columns=feature_cols)
        elif isinstance(samples, dict):
            X_in = pd.DataFrame([samples], columns=feature_cols)
        else:
            arr = np.asarray(samples)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            X_in = pd.DataFrame(arr, columns=feature_cols)
        preds = _as_1d_float(fitted_model.predict(X_in))
        return preds.tolist()

    return {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": "GeneratedTokens",
        "X_test": X_test,
        "y_test": _as_1d_float(y_test),
        "y_pred": y_pred,
        "predict_generated_tokens": predict_generated_tokens,
    }


def train_seq_pred_lr(df, test_size=0.2, random_state=42, **lr_params):

    required_cols = {"ContextTokens", "type", "GeneratedTokens"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame missing required columns: {missing}")
    if df.empty:
        raise ValueError("DataFrame is empty")

    feature_cols = ["ContextTokens", "type"]
    X = df[feature_cols].copy()
    y = df["GeneratedTokens"].to_numpy(dtype=np.float64)

    preprocessor = ColumnTransformer(
        [
            ("context", StandardScaler(), ["ContextTokens"]),
            ("type", OneHotEncoder(handle_unknown="ignore"), ["type"]),
        ],
        remainder="drop",
    )
    model = Pipeline(
        [
            ("pre", preprocessor),
            ("reg", LinearRegression(**lr_params)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model.fit(X_train, y_train)

    y_pred = _as_1d_float(model.predict(X_test))

    def predict_generated_tokens(samples, fitted_model=model):
        if isinstance(samples, pd.DataFrame):
            X_in = samples[feature_cols]
        elif isinstance(samples, list) and samples and isinstance(samples[0], dict):
            X_in = pd.DataFrame(samples, columns=feature_cols)
        elif isinstance(samples, dict):
            X_in = pd.DataFrame([samples], columns=feature_cols)
        else:
            arr = np.asarray(samples)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            X_in = pd.DataFrame(arr, columns=feature_cols)
        preds = _as_1d_float(fitted_model.predict(X_in))
        return preds.tolist()

    return {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": "GeneratedTokens",
        "X_test": X_test,
        "y_test": _as_1d_float(y_test),
        "y_pred": y_pred,
        "predict_generated_tokens": predict_generated_tokens,
    }


def train_seq_pred_xgb(df, test_size=0.2, random_state=42, **xgb_params):
    try:
        from xgboost import XGBRegressor
    except ImportError as e:
        raise ImportError(
            "train_seq_pred_xgb requires xgboost. Install with: pip install xgboost"
        ) from e

    required_cols = {"ContextTokens", "type", "GeneratedTokens"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame missing required columns: {missing}")
    if df.empty:
        raise ValueError("DataFrame is empty")

    feature_cols = ["ContextTokens", "type"]
    X = df[feature_cols].copy()
    y = df["GeneratedTokens"].to_numpy(dtype=np.float64)

    preprocessor = ColumnTransformer(
        [
            ("context", StandardScaler(), ["ContextTokens"]),
            ("type", OneHotEncoder(handle_unknown="ignore"), ["type"]),
        ],
        remainder="drop",
    )
    reg_defaults = dict(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        objective=_XGB_OBJECTIVE_MSE,
    )
    reg_defaults.update(xgb_params)
    model = Pipeline(
        [
            ("pre", preprocessor),
            ("reg", XGBRegressor(**reg_defaults)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model.fit(X_train, y_train)

    y_pred = _as_1d_float(model.predict(X_test))

    def predict_generated_tokens(samples, fitted_model=model):
        if isinstance(samples, pd.DataFrame):
            X_in = samples[feature_cols]
        elif isinstance(samples, list) and samples and isinstance(samples[0], dict):
            X_in = pd.DataFrame(samples, columns=feature_cols)
        elif isinstance(samples, dict):
            X_in = pd.DataFrame([samples], columns=feature_cols)
        else:
            arr = np.asarray(samples)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            X_in = pd.DataFrame(arr, columns=feature_cols)
        preds = _as_1d_float(fitted_model.predict(X_in))
        return preds.tolist()

    return {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": "GeneratedTokens",
        "X_test": X_test,
        "y_test": _as_1d_float(y_test),
        "y_pred": y_pred,
        "predict_generated_tokens": predict_generated_tokens,
    }

def train_seq_pred_nn(
    df,
    test_size=0.2,
    random_state=42,
    hidden_units=64,
    hidden_units_2=32,
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    validation_split=0.1,
    verbose=0,
    npu_ids=None,
):
    required_cols = {"ContextTokens", "type", "GeneratedTokens"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame missing required columns: {missing}")
    if df.empty:
        raise ValueError("DataFrame is empty")

    if not (0.0 < validation_split < 1.0):
        raise ValueError(f"validation_split must be in (0, 1), got {validation_split!r}")


    feature_cols = ["ContextTokens", "type"]
    X = df[feature_cols].copy()
    y = df["GeneratedTokens"].to_numpy(dtype=np.float64)


    preprocessor = ColumnTransformer(
        [
            ("context", StandardScaler(), ["ContextTokens"]),
            ("type", OneHotEncoder(handle_unknown="ignore"), ["type"]),
        ],
        remainder="drop",
    )


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    def _sparse_or_dense_to_float32(a):
        if hasattr(a, "toarray"):
            a = a.toarray()
        return np.asarray(a, dtype=np.float32)

    if npu_ids is not None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset

        if len(npu_ids) != 1:
            raise ValueError(
                "train_seq_pred_nn only supports a single Ascend NPU for the PyTorch path; "
                "pass npu_ids with exactly one id, e.g. [0]."
            )
        try:
            import torch_npu
        except ImportError as e:
            raise ImportError(
                "Ascend NPU training requires torch_npu (install the CANN-matched wheel for your PyTorch)."
            ) from e
        if not torch.npu.is_available():
            raise RuntimeError(
                "npu_ids was set but no Ascend NPU is visible (torch.npu.is_available() is False)."
            )
        npu_idx = int(npu_ids[0])
        torch.npu.set_device(npu_idx)
        device = torch.device(f"npu:{npu_idx}")

        X_all = _sparse_or_dense_to_float32(X_train_preprocessed)
        y_all = y_train.astype(np.float32)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_all, y_all, test_size=validation_split, random_state=random_state
        )

        in_dim = X_tr.shape[1]

        class _SeqPredNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(in_dim, hidden_units)
                self.fc2 = nn.Linear(hidden_units, hidden_units_2)
                self.out = nn.Linear(hidden_units_2, 1)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.out(x).squeeze(-1)

        torch.manual_seed(random_state)
        model = _SeqPredNN().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_ds = TensorDataset(
            torch.from_numpy(X_tr), torch.from_numpy(y_tr)
        )
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        train_losses = []
        val_losses = []
        best_state = None
        best_val = float("inf")

        for _epoch in range(epochs):
            model.train()
            tr_total = 0.0
            tr_n = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                tr_total += loss.item() * xb.size(0)
                tr_n += xb.size(0)
            train_losses.append(tr_total / max(tr_n, 1))

            model.eval()
            va_total = 0.0
            va_n = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    va_total += loss.item() * xb.size(0)
                    va_n += xb.size(0)
            va = va_total / max(va_n, 1)
            val_losses.append(va)

            if va < best_val:
                best_val = va
                best_state = copy.deepcopy(model.state_dict())

            if verbose:
                print(
                    f"epoch {_epoch + 1}/{epochs}: "
                    f"loss={train_losses[-1]:.6f} val_loss={val_losses[-1]:.6f}"
                )

        if best_state is not None:
            model.load_state_dict(best_state)

        best_idx = int(np.argmin(val_losses))
        best_epoch = best_idx + 1
        best_val_loss = float(val_losses[best_idx])

        fit_history = {"loss": train_losses, "val_loss": val_losses}

        X_test_t = torch.from_numpy(_sparse_or_dense_to_float32(X_test_preprocessed)).to(device)
        model.eval()
        with torch.no_grad():
            y_pred = _as_1d_float(model(X_test_t).detach().cpu().numpy())

        def predict_generated_tokens(
            samples,
            fitted_model=model,
            preprocessor=preprocessor,
            dev=device,
        ):
            if isinstance(samples, pd.DataFrame):
                X_in = samples[feature_cols]
            elif isinstance(samples, list) and samples and isinstance(samples[0], dict):
                X_in = pd.DataFrame(samples, columns=feature_cols)
            elif isinstance(samples, dict):
                X_in = pd.DataFrame([samples], columns=feature_cols)
            else:
                arr = np.asarray(samples)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                X_in = pd.DataFrame(arr, columns=feature_cols)

            X_in_preprocessed = preprocessor.transform(X_in)
            x = torch.from_numpy(_sparse_or_dense_to_float32(X_in_preprocessed)).to(dev)
            fitted_model.eval()
            with torch.no_grad():
                preds = fitted_model(x)
            return _as_1d_float(preds.detach().cpu().numpy()).tolist()

        return {
            "model": model,
            "feature_cols": feature_cols,
            "target_col": "GeneratedTokens",
            "X_test": X_test,
            "y_test": _as_1d_float(y_test),
            "y_pred": y_pred,
            "predict_generated_tokens": predict_generated_tokens,
            "fit_history": fit_history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "device": device,
            "npu_ids": list(npu_ids),
        }


    model = Sequential()
    model.add(Dense(hidden_units, input_dim=X_train_preprocessed.shape[1], activation="relu"))
    model.add(Dense(hidden_units_2, activation="relu"))
    model.add(Dense(1))


    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

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
            X_train_preprocessed,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
        )
        model.load_weights(ckpt_path)
    finally:
        try:
            os.remove(ckpt_path)
        except OSError:
            pass

    val_losses = history.history["val_loss"]
    best_idx = int(np.argmin(val_losses))
    best_epoch = best_idx + 1
    best_val_loss = float(val_losses[best_idx])

    y_pred = _as_1d_float(model.predict(X_test_preprocessed, verbose=verbose))


    def predict_generated_tokens(samples, fitted_model=model, preprocessor=preprocessor):
        if isinstance(samples, pd.DataFrame):
            X_in = samples[feature_cols]
        elif isinstance(samples, list) and samples and isinstance(samples[0], dict):
            X_in = pd.DataFrame(samples, columns=feature_cols)
        elif isinstance(samples, dict):
            X_in = pd.DataFrame([samples], columns=feature_cols)
        else:
            arr = np.asarray(samples)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            X_in = pd.DataFrame(arr, columns=feature_cols)


        X_in_preprocessed = preprocessor.transform(X_in)


        preds = _as_1d_float(fitted_model.predict(X_in_preprocessed))
        return preds.tolist()

    return {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": "GeneratedTokens",
        "X_test": X_test,
        "y_test": _as_1d_float(y_test),
        "y_pred": y_pred,
        "predict_generated_tokens": predict_generated_tokens,
        "fit_history": dict(history.history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "device": None,
        "npu_ids": None,
    }


def train_bert(
    df,
    text_col=None,
    target_col="GeneratedTokens",
    test_size=0.2,
    random_state=42,
    num_epochs=6,
    batch_size=16,
    learning_rate=1e-5,
    bert_pretrained_name="bert-base-uncased",
    freeze_bert_after_epoch=3,
    loss="mse",
    max_length=512,
    verbose=False,
    npu_ids=None,
):
    if target_col not in df.columns:
        raise KeyError(f"df must contain target column {target_col!r}")

    use_pre_tokenized = text_col is None
    if use_pre_tokenized:
        if "input_ids" not in df.columns or "attention_mask" not in df.columns:
            raise ValueError(
                "When text_col is None, df must include pre-tokenized columns 'input_ids' and "
                "'attention_mask' (as in latency_prediction / load_lmsys). "
                "Or pass text_col='prompt_content' (or another text column) for raw-text training."
            )
        feature_cols = ["input_ids", "attention_mask"]
        cols = [target_col, "input_ids", "attention_mask"]
    else:
        if text_col not in df.columns:
            raise KeyError(f"df must contain text column {text_col!r}")
        feature_cols = [text_col]
        cols = [text_col, target_col]

    if loss not in ("mse", "l1"):
        raise ValueError(f"loss must be 'mse' or 'l1', got {loss!r}")

    work = df[cols].dropna().copy()
    if len(work) < 10:
        raise ValueError("Need at least enough rows after dropna to train and evaluate.")

    trainval_df, test_df = train_test_split(
        work, test_size=test_size, random_state=random_state
    )
    train_df, val_df = train_test_split(
        trainval_df, test_size=0.25, random_state=random_state
    )

    tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_name)

    def _to_hf_pre(dframe):
        sub = dframe[["input_ids", "attention_mask", target_col]].copy()
        sub = sub.rename(columns={target_col: "labels"})
        sub["labels"] = sub["labels"].astype(np.float64)
        return Dataset.from_pandas(sub, preserve_index=False)

    def _to_hf_text(dframe, tcol):
        sub = dframe.rename(columns={tcol: "text", target_col: "labels"})
        ds = Dataset.from_pandas(sub[["text", "labels"]], preserve_index=False)

        def _tok(batch):
            enc = tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            enc["labels"] = [float(x) for x in batch["labels"]]
            return enc

        return ds.map(_tok, batched=True, remove_columns=["text", "labels"])

    if use_pre_tokenized:
        train_ds = _to_hf_pre(train_df)
        val_ds = _to_hf_pre(val_df)
        test_ds = _to_hf_pre(test_df)
    else:
        train_ds = _to_hf_text(train_df, text_col)
        val_ds = _to_hf_text(val_df, text_col)
        test_ds = _to_hf_text(test_df, text_col)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collator
    )

    npu_list = None
    use_npu_data_parallel = False
    if npu_ids is not None:
        try:
            import torch_npu
        except ImportError as e:
            raise ImportError(
                "Ascend NPU training requires torch_npu (install the CANN-matched wheel for your PyTorch)."
            ) from e
        if not torch.npu.is_available():
            raise RuntimeError(
                "npu_ids was set but no Ascend NPU is visible (torch.npu.is_available() is False)."
            )
        _seen = set()
        npu_list = []
        for x in npu_ids:
            i = int(x)
            if i not in _seen:
                _seen.add(i)
                npu_list.append(i)
        if not npu_list:
            raise ValueError("npu_ids must be a non-empty sequence of device indices.")
        torch.npu.set_device(npu_list[0])
        device = torch.device(f"npu:{npu_list[0]}")
        use_npu_data_parallel = len(npu_list) > 1
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class BertRegressionHead(nn.Module):
        def __init__(self, pretrained_name, hidden_dim=128):
            super().__init__()
            self.bert = BertModel.from_pretrained(pretrained_name)
            h = self.bert.config.hidden_size
            self.cls = nn.Linear(h, hidden_dim)
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 1)

        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            x = out.last_hidden_state[:, 0, :]
            x = F.relu(self.cls(x))
            x = F.relu(self.fc1(x))
            return self.fc2(x).squeeze(-1)

    model = BertRegressionHead(bert_pretrained_name).to(device)
    if use_npu_data_parallel:
        model = nn.DataParallel(model, device_ids=npu_list)

    if freeze_bert_after_epoch == 0:
        for p in model.bert.parameters():
            p.requires_grad = False
    elif freeze_bert_after_epoch is None:
        pass
    else:
        for p in model.bert.parameters():
            p.requires_grad = True

    criterion = nn.MSELoss() if loss == "mse" else nn.L1Loss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
    )

    history = {"loss": [], "val_loss": []}

    def _run_epoch(loader, train_mode):
        if train_mode:
            model.train()
        else:
            model.eval()
        total = 0.0
        n = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).float()
            if train_mode:
                optimizer.zero_grad()
            with torch.set_grad_enabled(train_mode):
                pred = model(input_ids=input_ids, attention_mask=attention_mask)
                ls = criterion(pred, labels)
                if train_mode:
                    ls.backward()
                    optimizer.step()
            total += ls.item() * labels.size(0)
            n += labels.size(0)
        return total / max(n, 1)

    for epoch in range(num_epochs):
        if (
            freeze_bert_after_epoch is not None
            and freeze_bert_after_epoch > 0
            and epoch == freeze_bert_after_epoch
        ):
            for p in model.bert.parameters():
                p.requires_grad = False
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
            )
        tr = _run_epoch(train_loader, True)
        va = _run_epoch(val_loader, False)
        history["loss"].append(tr)
        history["val_loss"].append(va)
        if verbose:
            print(f"epoch {epoch}: train_loss={tr:.6f} val_loss={va:.6f}")

    model.eval()
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pred = model(input_ids=input_ids, attention_mask=attention_mask)
            lab = batch["labels"]
            if hasattr(lab, "detach"):
                y_true_list.append(lab.detach().cpu().float().numpy())
            else:
                y_true_list.append(np.asarray(lab, dtype=np.float64))
            y_pred_list.append(pred.detach().cpu().float().numpy())
    y_test = _as_1d_float(np.concatenate(y_true_list))
    y_pred = _as_1d_float(np.concatenate(y_pred_list))

    def predict_generated_tokens(
        samples,
        fitted_model=model,
        tok=tokenizer,
        dev=device,
        max_len=max_length,
    ):
        if isinstance(samples, str):
            texts = [samples]
        elif isinstance(samples, list) and samples and isinstance(samples[0], str):
            texts = samples
        else:
            raise TypeError("samples must be a string or a list of strings")
        enc = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(dev) for k, v in enc.items()}
        fitted_model.eval()
        with torch.no_grad():
            out = fitted_model(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                token_type_ids=enc.get("token_type_ids"),
            )
        return _as_1d_float(out.detach().cpu().numpy()).tolist()

    if use_pre_tokenized:
        xcols = [c for c in ("input_ids", "attention_mask", "type") if c in test_df.columns]
        X_test = test_df[xcols].reset_index(drop=True)
    else:
        X_test = test_df[[text_col]].reset_index(drop=True)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "predict_generated_tokens": predict_generated_tokens,
        "fit_history": history,
        "device": device,
        "pre_tokenized": use_pre_tokenized,
        "npu_ids": None if npu_ids is None else list(npu_ids),
    }

