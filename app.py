import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from io import BytesIO
import itertools

st.title("Echo State Network Predictor (ESNapp)")

# -----------------------------
# 1️⃣ Upload training file
# -----------------------------
train_file = st.file_uploader("Upload Training Data (Excel or CSV)", type=["xlsx", "csv"])
model_trained = False

if train_file:
    file_type = train_file.name.split('.')[-1].lower()
    if file_type == 'csv':
        train_df = pd.read_csv(train_file)
    else:
        excel_sheets = pd.ExcelFile(train_file).sheet_names
        selected_sheet = st.selectbox("Select sheet for training data", excel_sheets)
        train_df = pd.read_excel(train_file, sheet_name=selected_sheet)

    st.subheader("Training Data Preview")
    st.dataframe(train_df.head())

    input_cols_train = st.multiselect(
        "Select input columns for training", 
        train_df.columns.tolist(), 
        default=train_df.columns[:-1]
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
    spectral_radius = st.slider("Spectral radius", 0.1, 2.0, 0.9, 0.05)
    n_reservoir = st.slider("Reservoir size", 10, 500, 50, 10)
    seed = st.number_input("Random seed", value=0)
    np.random.seed(seed)

    # -----------------------------
    # 2a️⃣ Optional Grid Search
    # -----------------------------
    grid_search = st.checkbox("Use Grid Search for best n_reservoir & spectral_radius?")

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

        # -----------------------------
        # Function to train and evaluate ESN
        # -----------------------------
        def train_esn(n_res, sr):
            Win = (np.random.rand(n_res, n_inputs) * 2 - 1) * 0.1
            W   = np.random.rand(n_res, n_res) * 2 - 1
            W   = W * (sr / max(abs(eig(W)[0])))
            x = np.zeros((n_res,1))
            X_res = []
            for t in range(X_train_norm.shape[1]):
                u = X_train_norm[:,t].reshape(-1,1)
                x = np.tanh(Win @ u + W @ x)
                X_res.append(x)
            X_res = np.hstack(X_res)
            reg = 1e-6
            Wout = np.linalg.solve(X_res @ X_res.T + reg*np.eye(n_res), X_res @ y_train_norm.T)
            return Win, W, Wout

        # -----------------------------
        # Grid Search
        # -----------------------------
        if grid_search:
            st.info("Performing grid search... This may take time for large reservoirs.")
            best_r2 = -np.inf
            best_params = {}
            # example ranges, user can modify later
            reservoir_options = [20, 50, 100]
            spectral_options = [0.7, 0.9, 1.1]
            for n_res_try, sr_try in itertools.product(reservoir_options, spectral_options):
                Win_try, W_try, Wout_try = train_esn(n_res_try, sr_try)
                x = np.zeros((n_res_try,1))
                X_res_test = []
                for t in range(X_train_norm.shape[1]):
                    u = X_train_norm[:,t].reshape(-1,1)
                    x = np.tanh(Win_try @ u + W_try @ x)
                    X_res_test.append(x)
                X_res_test = np.hstack(X_res_test)
                y_pred_norm = (Wout_try.T @ X_res_test).T
                y_pred = y_pred_norm * output_std + output_mean
                r2 = r2_score(y_train.flatten(), y_pred.flatten())
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = {'n_reservoir': n_res_try, 'spectral_radius': sr_try, 'Win': Win_try, 'W': W_try, 'Wout': Wout_try}
            st.success(f"Best grid search R²: {best_r2:.4f} | n_reservoir: {best_params['n_reservoir']}, spectral_radius: {best_params['spectral_radius']}")
            Win, W, Wout = best_params['Win'], best_params['W'], best_params['Wout']
            n_reservoir = best_params['n_reservoir']
        else:
            Win, W, Wout = train_esn(n_reservoir, spectral_radius)

        # Store model in session
        st.session_state['trained_model'] = {
            'Win': Win,
            'W': W,
            'Wout': Wout,
            'input_mean': input_mean,
            'input_std': input_std,
            'output_mean': output_mean,
            'output_std': output_std,
            'input_cols': input_cols_train,
            'output_cols': output_cols_train,
            'n_reservoir': n_reservoir
        }
        model_trained = True
        st.success("✅ ESN Model Trained Successfully!")

# -----------------------------
# 4️⃣ Upload Test File
# -----------------------------
test_file  = st.file_uploader("Upload Test Data (Excel or CSV)", type=["xlsx", "csv"], key="test_uploader")

if test_file and 'trained_model' in st.session_state:
    file_type_test = test_file.name.split('.')[-1].lower()
    if file_type_test == 'csv':
        test_df = pd.read_csv(test_file)
    else:
        excel_sheets_test = pd.ExcelFile(test_file).sheet_names
        selected_sheet_test = st.selectbox("Select sheet for test data", excel_sheets_test)
        test_df = pd.read_excel(test_file, sheet_name=selected_sheet_test)

    st.subheader("Test Data Preview")
    st.dataframe(test_df.head())

    input_cols_test = st.multiselect(
        "Select input columns for testing", 
        test_df.columns.tolist(), 
        default=[col for col in st.session_state['trained_model']['input_cols'] if col in test_df.columns.tolist()]
    )

    if st.button("Predict on Test File"):
        model = st.session_state['trained_model']

        X_test = test_df[input_cols_test].values
        X_test_norm = (X_test - model['input_mean']) / model['input_std']
        X_test_norm = X_test_norm.T

        # Predict
        x = np.zeros((model['n_reservoir'],1))
        X_test_res = []
        for t in range(X_test_norm.shape[1]):
            u = X_test_norm[:,t].reshape(-1,1)
            x = np.tanh(model['Win'] @ u + model['W'] @ x)
            X_test_res.append(x)
        X_test_res = np.hstack(X_test_res)
        y_pred_norm = (model['Wout'].T @ X_test_res).T
        y_pred = y_pred_norm * model['output_std'] + model['output_mean']

        # -----------------------------
        # Show results
        # -----------------------------
        st.subheader("Predicted Output")
        pred_df = pd.DataFrame(y_pred, columns=model['output_cols'])
        st.dataframe(pred_df)

        # Plot predicted alone
        st.subheader("Predicted Values")
        fig1, ax1 = plt.subplots()
        ax1.plot(y_pred.flatten(), 'r--', label="Predicted")
        ax1.set_xlabel("Sample")
        ax1.set_ylabel("Output")
        ax1.legend()
        st.pyplot(fig1)

        # -----------------------------
        # Plot predicted vs actual if available
        # -----------------------------
        if test_df.shape[1] >= len(input_cols_test)+1:
            y_actual = test_df.iloc[:, len(input_cols_test)].values
            R2 = r2_score(y_actual, y_pred.flatten())
            st.write(f"R² score on test file: {R2:.4f}")

            st.subheader("Predicted vs Actual")
            fig2, ax2 = plt.subplots()
            ax2.plot(y_pred.flatten(), 'r--', label="Predicted")
            ax2.plot(y_actual, 'b', label="Actual")
            ax2.set_xlabel("Sample")
            ax2.set_ylabel("Output")
            ax2.legend()
            st.pyplot(fig2)

        # Download predictions
        output = BytesIO()
        pred_df.to_excel(output, index=False, engine='openpyxl')
        st.download_button(
            label="Download Predictions as Excel",
            data=output.getvalue(),
            file_name="ESN_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
