import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from io import BytesIO

st.title("Echo State Network Predictor")

# -----------------------------
# 1️⃣ Upload training and test files
# -----------------------------
train_file = st.file_uploader("Upload Training Excel (Inputs + Outputs)", type=["xlsx"])
test_file  = st.file_uploader("Upload Test Excel (Inputs only)", type=["xlsx"])

if train_file and test_file:
    train_df = pd.read_excel(train_file)
    test_df  = pd.read_excel(test_file)

    st.subheader("Training Data Preview")
    st.dataframe(train_df.head())
    st.subheader("Test Data Preview")
    st.dataframe(test_df.head())

    # User selects input/output columns
    input_cols_train = st.multiselect("Select input columns for training", train_df.columns.tolist(), default=train_df.columns[:-1])
    output_cols_train = st.multiselect("Select output column(s) for training", train_df.columns.tolist(), default=[train_df.columns[-1]])

    input_cols_test = st.multiselect("Select input columns for testing", test_df.columns.tolist(), default=input_cols_train)

    if input_cols_train and output_cols_train and input_cols_test:
        # -----------------------------
        # 2️⃣ Prepare data
        # -----------------------------
        X_train = train_df[input_cols_train].values
        y_train = train_df[output_cols_train].values
        X_test  = test_df[input_cols_test].values

        # Normalize
        input_mean = X_train.mean(axis=0)
        input_std  = X_train.std(axis=0)
        X_train_norm = (X_train - input_mean) / input_std
        X_test_norm  = (X_test - input_mean) / input_std

        output_mean = y_train.mean(axis=0)
        output_std  = y_train.std(axis=0)
        y_train_norm = (y_train - output_mean) / output_std

        # Transpose for ESN
        X_train_norm = X_train_norm.T
        y_train_norm = y_train_norm.T
        X_test_norm  = X_test_norm.T

        # -----------------------------
        # 3️⃣ ESN Hyperparameters
        # -----------------------------
        n_inputs = X_train_norm.shape[0]
        spectral_radius = st.slider("Spectral radius", min_value=0.1, max_value=2.0, value=0.9, step=0.05)
        n_reservoir = st.slider("Reservoir size", min_value=10, max_value=500, value=50, step=10)
        seed = st.number_input("Random seed", value=0)
        np.random.seed(seed)

        # -----------------------------
        # 4️⃣ Initialize ESN
        # -----------------------------
        Win = (np.random.rand(n_reservoir, n_inputs) * 2 - 1) * 0.1
        W   = np.random.rand(n_reservoir, n_reservoir) * 2 - 1
        W   = W * (spectral_radius / max(abs(eig(W)[0])))

        # -----------------------------
        # 5️⃣ Collect reservoir states for training
        # -----------------------------
        x = np.zeros((n_reservoir,1))
        X_res = []

        for t in range(X_train_norm.shape[1]):
            u = X_train_norm[:,t].reshape(-1,1)
            x = np.tanh(Win @ u + W @ x)
            X_res.append(x)

        X_res = np.hstack(X_res)

        # Ridge regression
        reg = 1e-6
        Wout = np.linalg.solve(X_res @ X_res.T + reg*np.eye(n_reservoir), X_res @ y_train_norm.T)

        # -----------------------------
        # 6️⃣ Predict on test set
        # -----------------------------
        x = np.zeros((n_reservoir,1))
        X_test_res = []

        for t in range(X_test_norm.shape[1]):
            u = X_test_norm[:,t].reshape(-1,1)
            x = np.tanh(Win @ u + W @ x)
            X_test_res.append(x)

        X_test_res = np.hstack(X_test_res)
        y_pred_norm = (Wout.T @ X_test_res).T

        # Denormalize
        y_pred = y_pred_norm * output_std + output_mean

        # -----------------------------
        # 7️⃣ Show results
        # -----------------------------
        st.subheader("Predicted Output")
        pred_df = pd.DataFrame(y_pred, columns=output_cols_train)
        st.dataframe(pred_df)

        # Compute R² if actual test output is available
        if len(output_cols_train) == 1 and test_df.shape[1] >= len(input_cols_test)+1:
            y_test_actual = test_df.iloc[:, len(input_cols_test)].values
            R2 = r2_score(y_test_actual, y_pred.flatten())
            st.write(f"R² score: {R2:.4f}")

        # Plot results
        st.subheader("Prediction Plot")
        fig, ax = plt.subplots()
        ax.plot(y_pred.flatten(), 'r--', label="Predicted")
        if len(output_cols_train) == 1 and test_df.shape[1] >= len(input_cols_test)+1:
            ax.plot(y_test_actual, 'b', label="Actual")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Output")
        ax.legend()
        st.pyplot(fig)

        # -----------------------------
        # 8️⃣ Download predictions as Excel
        # -----------------------------
        def convert_df_to_excel(df):
            output = BytesIO()
            df.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)
            return output

        st.download_button(
            label="Download Predictions as Excel",
            data=convert_df_to_excel(pred_df),
            file_name="ESN_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
