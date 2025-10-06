import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from io import BytesIO
import itertools

st.set_page_config(layout="wide")
st.title("Echo State Network Predictor (ESNapp)")

# -----------------------------
# 1️⃣ Upload training file
# -----------------------------
file_uploader = st.file_uploader("Upload Training Data (Excel or CSV)", type=["xlsx", "csv"])
model_trained = False

if file_uploader:
    # Handle both Excel and CSV files
    file_type = file_uploader.name.split('.')[-1]
    if file_type == 'csv':
        train_df = pd.read_csv(file_uploader)
    elif file_type == 'xlsx':
        excel_sheets = pd.ExcelFile(file_uploader).sheet_names
        selected_sheet = st.selectbox("Select the sheet for training data", excel_sheets)
        train_df = pd.read_excel(file_uploader, sheet_name=selected_sheet)
    
    st.subheader("Training Data Preview")
    st.dataframe(train_df.head())

    input_cols_train = st.multiselect(
        "Select input columns for training", 
        train_df.columns.tolist(), 
        default=train_df.columns[:-1].tolist()
    )
    output_cols_train = st.multiselect(
        "Select output column(s) for training", 
        train_df.columns.tolist(), 
        default=[train_df.columns[-1]]
    )

    # -----------------------------
    # 2️⃣ ESN Hyperparameters
    # -----------------------------
    st.subheader("ESN Hyperparameters")
    
    # Grid search option
    grid_search = st.checkbox("Use Grid Search for best hyperparameters?")
    if grid_search:
        n_reservoir_range = st.text_input("Reservoir sizes (comma-separated)", "50, 100, 200, 300")
        spectral_radius_range = st.text_input("Spectral radius values", "0.6, 0.8, 1.0, 1.2")
        leak_rate_range = st.text_input("Leak rate values", "0.1, 0.3, 0.5, 0.7")
        ridge_range = st.text_input("Ridge λ values", "1e-8, 1e-6, 1e-4, 1e-2")
        
        try:
            reservoir_options = [int(x.strip()) for x in n_reservoir_range.split(',')]
            spectral_options = [float(x.strip()) for x in spectral_radius_range.split(',')]
            leak_options = [float(x.strip()) for x in leak_rate_range.split(',')]
            ridge_options = [float(x.strip()) for x in ridge_range.split(',')]
        except ValueError:
            st.error("Invalid hyperparameter input.")
            st.stop()
    else:
        spectral_radius = st.slider("Spectral radius", 0.1, 2.0, 0.9, 0.05)
        n_reservoir = st.slider("Reservoir size", 10, 500, 200, 10)
        leak_rate = st.slider("Leak rate", 0.0, 1.0, 0.3, 0.05)
        ridge_lambda = st.number_input("Ridge regularization λ", value=1e-6, format="%.1e")

    seed = st.number_input("Random seed", value=0)
    np.random.seed(seed)

    # -----------------------------
    # 3️⃣ Train ESN Button
    # -----------------------------
    if st.button("Train ESN Model"):
        X_train = train_df[input_cols_train].values
        y_train = train_df[output_cols_train].values

        # Normalize
        input_mean = X_train.mean(axis=0)
        input_std  = X_train.std(axis=0)
        X_train_norm = (X_train - input_mean) / input_std

        output_mean = y_train.mean(axis=0)
        output_std  = y_train.std(axis=0)
        y_train_norm = (y_train - output_mean) / output_std

        # Transpose
        X_train_norm = X_train_norm.T
        y_train_norm = y_train_norm.T
        n_inputs = X_train_norm.shape[0]

        # Split into train/val for grid search
        n_samples = X_train_norm.shape[1]
        val_split = int(0.8 * n_samples)
        Xtr, Xval = X_train_norm[:, :val_split], X_train_norm[:, val_split:]
        Ytr, Yval = y_train_norm[:, :val_split], y_train_norm[:, val_split:]

        # -----------------------------
        # ESN training function
        # -----------------------------
        @st.cache_data
        def train_esn(n_res, sr, leak, ridge, Xtr, Ytr, Xval, Yval):
            np.random.seed(seed)
            Win = (np.random.rand(n_res, n_inputs) * 2 - 1) * 0.1
            W = np.random.rand(n_res, n_res) * 2 - 1
            W = W * (sr / max(abs(eig(W)[0])))

            def run_reservoir(X):
                x = np.zeros((n_res, 1))
                states = []
                for t in range(X.shape[1]):
                    u = X[:, t].reshape(-1, 1)
                    x = (1 - leak) * x + leak * np.tanh(Win @ u + W @ x)
                    states.append(x)
                return np.hstack(states)

            Xres_tr = run_reservoir(Xtr)
            Xres_val = run_reservoir(Xval)

            # Ridge regression
            I = np.eye(Xres_tr.shape[0])
            Wout = Ytr @ Xres_tr.T @ np.linalg.inv(Xres_tr @ Xres_tr.T + ridge * I)

            Yval_pred = Wout @ Xres_val
            r2 = r2_score(Yval.flatten(), Yval_pred.T.flatten())
            return Wout, Win, W, r2

        # -----------------------------
        # Grid Search
        # -----------------------------
        if grid_search:
            st.info("Running grid search...")
            best_r2 = -np.inf
            best_params = {}

            for nr, sr, leak, ridge in itertools.product(reservoir_options, spectral_options, leak_options, ridge_options):
                Wout, Win, W, r2 = train_esn(nr, sr, leak, ridge, Xtr, Ytr, Xval, Yval)
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = dict(nr=nr, sr=sr, leak=leak, ridge=ridge, Win=Win, W=W, Wout=Wout)

            if best_r2 <= 0:
                st.warning(f"Best validation R² is low: {best_r2:.4f}. Try wider hyperparameter ranges.")
            else:
                st.success(f"Best hyperparams → R² (val): {best_r2:.4f}, "
                           f"n_res={best_params['nr']}, sr={best_params['sr']}, leak={best_params['leak']}, ridge={best_params['ridge']}")
            params = best_params

        else:
            Wout, Win, W, r2_val = train_esn(n_reservoir, spectral_radius, leak_rate, ridge_lambda, Xtr, Ytr, Xval, Yval)
            st.success(f"Model trained → Validation R²: {r2_val:.4f}")
            params = dict(nr=n_reservoir, sr=spectral_radius, leak=leak_rate, ridge=ridge_lambda, Win=Win, W=W, Wout=Wout)

        # Store
        st.session_state['trained_model'] = dict(
            Win=params['Win'],
            W=params['W'],
            Wout=params['Wout'],
            n_reservoir=params['nr'],
            leak=params['leak'],
            input_mean=input_mean,
            input_std=input_std,
            output_mean=output_mean,
            output_std=output_std,
            input_cols=input_cols_train,
            output_cols=output_cols_train
        )

# -----------------------------
# 4️⃣ Testing
# -----------------------------
test_file = st.file_uploader("Upload Test Data", type=["xlsx", "csv"], key="test_uploader")
if test_file and 'trained_model' in st.session_state:
    if test_file.name.endswith("csv"):
        test_df = pd.read_csv(test_file)
    else:
        excel_sheets_test = pd.ExcelFile(test_file).sheet_names
        selected_sheet_test = st.selectbox("Select test sheet", excel_sheets_test)
        test_df = pd.read_excel(test_file, sheet_name=selected_sheet_test)

    st.subheader("Test Data Preview")
    st.dataframe(test_df.head())

    input_cols_test = st.multiselect("Select input columns for testing",
                                     test_df.columns.tolist(),
                                     default=st.session_state['trained_model']['input_cols'])

    if st.button("Predict on Test File"):
        model = st.session_state['trained_model']

        # Normalize
        X_test = test_df[input_cols_test].values
        X_test_norm = (X_test - model['input_mean']) / model['input_std']
        X_test_norm = X_test_norm.T

        # Reservoir forward
        x = np.zeros((model['n_reservoir'], 1))
        Xres_test = []
        for t in range(X_test_norm.shape[1]):
            u = X_test_norm[:, t].reshape(-1, 1)
            x = (1 - model['leak']) * x + model['leak'] * np.tanh(model['Win'] @ u + model['W'] @ x)
            Xres_test.append(x)
        Xres_test = np.hstack(Xres_test)

        y_pred_norm = (model['Wout'] @ Xres_test).T
        y_pred = y_pred_norm * model['output_std'] + model['output_mean']

        # Results
        st.subheader("Predicted Output")
        pred_df = pd.DataFrame(y_pred, columns=model['output_cols'])
        st.dataframe(pred_df)

        # Plot
        if any(col in test_df.columns for col in model['output_cols']):
            y_actual = test_df[model['output_cols']].values
            R2_test = r2_score(y_actual, y_pred)
            st.success(f"R² on Test Data: {R2_test:.4f}")

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(y_actual, 'b', label="Actual")
            ax.plot(y_pred, 'r--', label="Predicted")
            ax.set_title("Predicted vs Actual")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(y_pred, 'r', label="Predicted")
            ax.set_title("Predicted Output")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        # Download
        output = BytesIO()
        pred_df.to_excel(output, index=False, engine='openpyxl')
        st.download_button("Download Predictions", data=output.getvalue(),
                           file_name="ESN_predictions.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
