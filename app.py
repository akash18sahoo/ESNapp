# app.py - Improved ESNapp (keeps all features + better generalization)
import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from io import BytesIO
import itertools

st.set_page_config(layout="wide", page_title="ESN Predictor")
st.title("Echo State Network Predictor (ESNapp) — Improved Generalization")

# -----------------------------
# 0. Utility helpers
# -----------------------------
def make_lagged(X, n_lags):
    """Make lagged features: X shape (T, n_features) -> returns (T-n_lags, n_features*n_lags) """
    if n_lags <= 1:
        return X.copy()
    T, F = X.shape
    rows = T - (n_lags - 1)
    if rows <= 0:
        raise ValueError("n_lags too large for sequence length")
    out = np.zeros((rows, F * n_lags))
    for i in range(rows):
        window = X[i:i + n_lags, :].flatten(order='F')  # stack features for each lag
        out[i, :] = window
    return out

def compute_states(Win, W, Xn, leak, washout):
    """Compute reservoir states for normalized input Xn (shape n_inputs x T)"""
    n_res = W.shape[0]
    T = Xn.shape[1]
    x = np.zeros((n_res,1))
    states = []
    for t in range(T):
        u = Xn[:, t].reshape(-1,1)
        x = (1 - leak) * x + leak * np.tanh(Win @ u + W @ x)
        if t >= washout:
            states.append(x)
    if len(states) == 0:
        return np.zeros((n_res,0))
    return np.hstack(states)  # shape (n_res, T-washout)

def train_readout_ridge(Xres, Y, ridge):
    """Wout = Y @ Xres.T @ inv(Xres @ Xres.T + ridge I)"""
    I = np.eye(Xres.shape[0])
    Wout = Y @ Xres.T @ np.linalg.inv(Xres @ Xres.T + ridge * I)
    return Wout

# -----------------------------
# 1️⃣ Upload training file
# -----------------------------
st.header("1) Training data")
file_uploader = st.file_uploader("Upload Training Data (Excel or CSV)", type=["xlsx","csv"])
if not file_uploader:
    st.info("Upload a training file (e.g., Heat6).")
# keep options even if not uploaded yet
model_key = "trained_model"

# UI options common
with st.expander("Options (advanced) — keep defaults if unsure", expanded=False):
    st.markdown("**Grid search / regularization / robustness options**")
    default_grid = st.checkbox("Enable grid search (recommended)", value=True)
    seed_count = st.number_input("Seeds per grid point (ensemble)", min_value=1, max_value=10, value=3)
    washout = st.number_input("Washout steps (skip initial transient)", min_value=0, max_value=200, value=5)
    augment_noise = st.checkbox("Use small Gaussian augmentation on training inputs", value=False)
    augment_sigma = st.number_input("Augmentation σ (relative to input std)", value=0.01, format="%.4f")

# keep UI consistent when file not uploaded
if file_uploader:
    # load training data
    if file_uploader.name.lower().endswith(".csv"):
        train_df = pd.read_csv(file_uploader)
    else:
        sheets = pd.ExcelFile(file_uploader).sheet_names
        sheet = st.selectbox("Select sheet for training data", sheets)
        train_df = pd.read_excel(file_uploader, sheet_name=sheet)

    st.subheader("Training preview")
    st.dataframe(train_df.head())

    # select inputs/outputs and optional lagging
    input_cols = st.multiselect("Select input columns (training)", train_df.columns.tolist(), default=train_df.columns[:-1].tolist())
    output_cols = st.multiselect("Select output column(s) (training)", train_df.columns.tolist(), default=[train_df.columns[-1]])
    n_lags = st.number_input("Number of time-lag steps to include as inputs (1 = no lag)", min_value=1, max_value=50, value=1)

    # Hyperparameter UI
    st.subheader("Hyperparameters (training)")
    if default_grid:
        reservoir_range = st.text_input("Reservoir sizes (comma-separated)", "50,100,200")
        sr_range = st.text_input("Spectral radii (comma-separated)", "0.6,0.8,1.0")
        leak_range = st.text_input("Leak rates (comma-separated)", "0.1,0.3,0.5")
        ridge_range = st.text_input("Ridge λ (comma-separated)", "1e-8,1e-6,1e-4")
        try:
            reservoir_options = [int(x.strip()) for x in reservoir_range.split(',')]
            sr_options = [float(x.strip()) for x in sr_range.split(',')]
            leak_options = [float(x.strip()) for x in leak_range.split(',')]
            ridge_options = [float(x.strip()) for x in ridge_range.split(',')]
        except Exception as e:
            st.error("Invalid grid input; please enter comma-separated numbers.")
            st.stop()
    else:
        spectral_radius = st.slider("Spectral radius", 0.1, 2.0, 0.9, 0.05)
        n_reservoir = st.slider("Reservoir size", 10, 1000, 200, 10)
        leak_rate = st.slider("Leak rate", 0.0, 1.0, 0.3, 0.05)
        ridge_lambda = st.number_input("Ridge λ", value=1e-6, format="%.1e")

    seed_base = st.number_input("Random seed base", value=0, step=1)

    # Train button
    if st.button("Train ESN Model"):
        # prepare input/target arrays (handle lags)
        if len(input_cols) == 0 or len(output_cols) == 0:
            st.error("Pick input and output columns.")
            st.stop()

        X_full = train_df[input_cols].values  # shape (T, n_in)
        Y_full = train_df[output_cols].values  # shape (T, n_out)

        # create lagged inputs if requested
        if n_lags > 1:
            try:
                X_lagged = make_lagged(X_full, n_lags)  # shape (T-n_lags+1, n_in*n_lags)
                Y_lagged = Y_full[n_lags-1:]  # align outputs
                X = X_lagged
                Y = Y_lagged
            except Exception as e:
                st.error("Error creating lagged features: " + str(e))
                st.stop()
        else:
            X = X_full.copy()
            Y = Y_full.copy()

        # normalize training set (keep stats)
        X_mean = X.mean(axis=0); X_std = X.std(axis=0) + 1e-12
        Y_mean = Y.mean(axis=0); Y_std = Y.std(axis=0) + 1e-12
        Xn = ((X - X_mean) / X_std).T   # shape (n_in_eff, T)
        Yn = ((Y - Y_mean) / Y_std).T   # shape (n_out, T)
        n_inputs_eff = Xn.shape[0]

        # train/validation split (time-based)
        T = Xn.shape[1]
        if T < 10:
            st.error("Not enough samples after lagging to train (need >= 10).")
            st.stop()
        split = int(0.7 * T)
        Xtr, Xval = Xn[:, :split], Xn[:, split:]
        Ytr, Yval = Yn[:, :split], Yn[:, split:]

        # optional augmentation: add small gaussian noise to training inputs
        if augment_noise:
            noise_sigma = augment_sigma
        else:
            noise_sigma = 0.0

        # ESN train function with washout, leak, ridge, returns Wout, Win, W
        def esn_train_eval(n_res, sr, leak, ridge, seeds_list):
            """Train multiple seeds and average validation R2. Return model of best seed (highest val R2) and mean val R2"""
            val_r2s = []
            seed_models = []
            for sd in seeds_list:
                np.random.seed(sd)
                Win = (np.random.rand(n_res, n_inputs_eff) * 2 - 1) * 0.1
                W = np.random.rand(n_res, n_res) * 2 - 1
                W = W * (sr / max(abs(eig(W)[0])))

                # optionally add noise augmentation on training inputs (random each seed)
                if noise_sigma > 0.0:
                    Xtr_aug = Xtr + np.random.normal(scale=noise_sigma, size=Xtr.shape)
                else:
                    Xtr_aug = Xtr

                # collect states (skip washout)
                Xres_tr = compute_states(Win, W, Xtr_aug, leak, washout)  # (n_res, Ttr-washout)
                Ytr_cut = Ytr[:, washout:] if Xres_tr.shape[1] == Ytr.shape[1]-washout else Ytr[:, :Xres_tr.shape[1]]
                # If lengths mismatch, align by truncation
                if Xres_tr.shape[1] != Ytr.shape[1] - washout:
                    # attempt safe alignment: take minimum
                    L = min(Xres_tr.shape[1], Ytr.shape[1])
                    Xres_tr = Xres_tr[:, :L]
                    Ytr_cut = Ytr[:, :L]

                # ridge readout
                Wout = train_readout_ridge(Xres_tr, Ytr_cut, ridge)

                # validation prediction
                Xres_val = compute_states(Win, W, Xval, leak, washout)
                if Xres_val.shape[1] == 0:
                    val_r2 = -np.inf
                else:
                    # align Yval shape
                    Yval_cut = Yval[:, :Xres_val.shape[1]]
                    yval_pred = (Wout @ Xres_val).T * Y_std + Y_mean  # shape (Tval, n_out)
                    yval_true = (Yval_cut.T * Y_std) + Y_mean
                    try:
                        val_r2 = r2_score(yval_true.flatten(), yval_pred.flatten())
                    except Exception:
                        val_r2 = -np.inf

                val_r2s.append(val_r2)
                seed_models.append((val_r2, Win, W, Wout))

            # choose model of best seed (max val_r2) and compute mean val r2
            mean_val = np.mean([v for v in val_r2s if np.isfinite(v)]) if len(val_r2s)>0 else -np.inf
            best_seed_model = max(seed_models, key=lambda x: x[0])
            return mean_val, best_seed_model  # (mean_val_r2, (val_r2, Win, W, Wout))

        # run grid search or single-train
        if default_grid:
            st.info("Running robust grid search (validation-based, multi-seed). This may take a while...")
            total = len(reservoir_options)*len(sr_options)*len(leak_options)*len(ridge_options)
            progress = st.progress(0)
            best_overall = (-np.inf, None)  # best (mean_val_r2, modeltuple)
            i = 0
            for (nr, sr, leak, ridge) in itertools.product(reservoir_options, sr_options, leak_options, ridge_options):
                i += 1
                progress.progress(i/total)
                # seeds list
                seeds_list = [int(seed_base + s) for s in range(seed_count)]
                mean_val_r2, best_seed_model = esn_train_eval(nr, sr, leak, ridge, seeds_list)
                if mean_val_r2 > best_overall[0]:
                    best_overall = (mean_val_r2, (nr, sr, leak, ridge, best_seed_model))
            progress.empty()

            best_mean_r2, best_info = best_overall
            if best_info is None:
                st.error("Grid search failed to find a workable model.")
                st.stop()
            nr_best, sr_best, leak_best, ridge_best, seed_model = best_info
            val_r2_of_best_seed, Win_best, W_best, Wout_best = seed_model

            st.success(f"Selected hyperparams (mean val R² over seeds = {best_mean_r2:.4f}): "
                       f"n_res={nr_best}, sr={sr_best}, leak={leak_best}, ridge={ridge_best} (best-seed val R²={val_r2_of_best_seed:.4f})")

            # store trained model and normalization
            st.session_state[model_key] = dict(
                Win=Win_best, W=W_best, Wout=Wout_best, n_reservoir=nr_best, leak=leak_best,
                input_cols=input_cols, output_cols=output_cols,
                input_mean=X_mean, input_std=X_std, output_mean=Y_mean, output_std=Y_std,
                n_lags=n_lags
            )
            model_trained = True

        else:
            # single config train with ensemble seeds -> pick best seed by val r2
            seeds_list = [int(seed_base + s) for s in range(seed_count)]
            mean_val_r2, best_seed_model = esn_train_eval(n_reservoir, spectral_radius, leak_rate, ridge_lambda, seeds_list)
            val_r2_of_best_seed, Win_best, W_best, Wout_best = best_seed_model
            st.success(f"Trained model (val R² of chosen seed = {val_r2_of_best_seed:.4f})")

            st.session_state[model_key] = dict(
                Win=Win_best, W=W_best, Wout=Wout_best, n_reservoir=n_reservoir, leak=leak_rate,
                input_cols=input_cols, output_cols=output_cols,
                input_mean=X_mean, input_std=X_std, output_mean=Y_mean, output_std=Y_std,
                n_lags=n_lags
            )
            model_trained = True

# -----------------------------
# 4️⃣ Upload test file & predict
# -----------------------------
st.header("2) Test / Predict")
test_file = st.file_uploader("Upload Test Data (Excel or CSV) — you can upload multiple different test files", type=["xlsx","csv"], key="test_uploader")
use_test_norm = st.checkbox("Normalize test data using its own mean/std (toggle)", value=False)

if test_file and model_key in st.session_state:
    # load test file
    if test_file.name.lower().endswith(".csv"):
        test_df = pd.read_csv(test_file)
    else:
        sheets = pd.ExcelFile(test_file).sheet_names
        sheet_t = st.selectbox("Select test sheet", sheets)
        test_df = pd.read_excel(test_file, sheet_name=sheet_t)

    st.subheader("Test preview")
    st.dataframe(test_df.head())

    # choose input cols for test
    default_test_cols = [c for c in st.session_state[model_key]['input_cols'] if c in test_df.columns.tolist()]
    input_test_cols = st.multiselect("Select input columns for testing", test_df.columns.tolist(), default=default_test_cols)

    if st.button("Predict on Test File"):
        mdl = st.session_state[model_key]
        if set(input_test_cols) != set(mdl['input_cols']):
            st.error("Selected test input columns must match the training input columns (including order).")
            st.stop()

        # construct X_test considering lags
        Xfull_test = test_df[input_test_cols].values
        if mdl.get('n_lags',1) > 1:
            try:
                X_test_proc = make_lagged(Xfull_test, mdl['n_lags'])
            except Exception as e:
                st.error("Error creating lagged features for test: " + str(e))
                st.stop()
        else:
            X_test_proc = Xfull_test.copy()

        # normalize test either by training stats or its own stats
        if use_test_norm:
            t_mean = X_test_proc.mean(axis=0); t_std = X_test_proc.std(axis=0) + 1e-12
            Xn_test = ((X_test_proc - t_mean) / t_std).T
        else:
            Xn_test = ((X_test_proc - mdl['input_mean']) / mdl['input_std']).T

        # forward through reservoir with washout (the same washout used during training)
        n_res = mdl['n_reservoir']
        leak = mdl.get('leak', 0.3)
        x = np.zeros((n_res,1))
        states_test = []
        for t in range(Xn_test.shape[1]):
            u = Xn_test[:, t].reshape(-1,1)
            x = (1 - leak) * x + leak * np.tanh(mdl['Win'] @ u + mdl['W'] @ x)
            # do NOT remove washout here: we predict for every step; if training used washout, outputs correspond accordingly.
            states_test.append(x)
        if len(states_test) == 0:
            st.error("No test steps to predict after processing.")
            st.stop()
        Xres_test = np.hstack(states_test)  # (n_res, Ttest)

        ypred_norm = (mdl['Wout'] @ Xres_test).T  # shape Ttest x n_out
        ypred = ypred_norm * mdl['output_std'] + mdl['output_mean']

        # show predictions
        st.subheader("Predicted Output")
        pred_df = pd.DataFrame(ypred, columns=mdl['output_cols'])
        st.dataframe(pred_df)

        # compute R2 if actuals present
        if all(col in test_df.columns for col in mdl['output_cols']):
            y_true = test_df[mdl['output_cols']].values
            # align lengths if lagging truncated test
            if y_true.shape[0] != ypred.shape[0]:
                # align by using last min length
                L = min(y_true.shape[0], ypred.shape[0])
                y_true = y_true[-L:]
                ypred = ypred[-L:]
            R2 = r2_score(y_true.flatten(), ypred.flatten())
            st.success(f"R² on test file: {R2:.4f}")
            # plot predicted vs actual
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(y_true, 'b', label='Actual', linewidth=2)
            ax.plot(ypred, 'r--', label='Predicted', linewidth=2)
            ax.set_title("Predicted vs Actual")
            ax.legend(); ax.grid(True)
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(ypred, 'r', label='Predicted', linewidth=2)
            ax.set_title("Predicted Output")
            ax.legend(); ax.grid(True)
            st.pyplot(fig)

        # download
        out = BytesIO()
        pred_df.to_excel(out, index=False, engine='openpyxl')
        out.seek(0)
        st.download_button("Download Predictions", data=out.getvalue(),
                           file_name="ESN_predictions.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    if 'trained_model' not in st.session_state:
        st.info("First: upload training file and train model. Then upload test file here.")
