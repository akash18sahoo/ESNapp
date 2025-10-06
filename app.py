import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from io import BytesIO

st.title("🔵 ESN App – Stable Version with Ridge Regression")

# ------------------------------
# Utility functions
# ------------------------------

def scale_data(train_df, test_df, input_cols, target_col):
    """Scale inputs and target to [0,1] for ESN stability"""
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    # Scale input columns
    for col in input_cols:
        min_val, max_val = train_df[col].min(), train_df[col].max()
        train_scaled[col] = (train_df[col] - min_val) / (max_val - min_val + 1e-8)
        test_scaled[col] = (test_df[col] - min_val) / (max_val - min_val + 1e-8)

    # Scale target
    min_y, max_y = train_df[target_col].min(), train_df[target_col].max()
    train_scaled[target_col] = (train_df[target_col] - min_y) / (max_y - min_y + 1e-8)

    return train_scaled, test_scaled, (min_y, max_y)


def inverse_scale(y_scaled, y_min, y_max):
    return y_scaled * (y_max - y_min) + y_min


def generate_reservoir(n_reservoir, spectral_radius, n_inputs):
    """Generate random reservoir matrix and scale by spectral radius"""
    W_in = (np.random.rand(n_reservoir, n_inputs + 1) - 0.5)  # +1 for bias
    W = np.random.rand(n_reservoir, n_reservoir) - 0.5
    # Scale to desired spectral radius
    eigvals = np.linalg.eigvals(W)
    W *= spectral_radius / (np.max(np.abs(eigvals)) + 1e-8)
    return W_in, W


def train_esn(X, y, n_reservoir, spectral_radius, ridge_alpha, washout):
    """
    Train ESN using ridge regression
    X: input features (n_inputs x T)
    y: target (1 x T)
    """
    n_inputs, T = X.shape
    W_in, W = generate_reservoir(n_reservoir, spectral_radius, n_inputs)

    # Collect reservoir states
    states = np.zeros((n_reservoir, T))
    x = np.zeros((n_reservoir, 1))
    for t in range(T):
        u = np.vstack([1, X[:, t:t+1]])  # add bias
        x = np.tanh(W_in @ u + W @ x)
        states[:, t] = x[:, 0]

    # Apply washout
    states_w = states[:, washout:]
    y_w = y[washout:]

    # Ridge regression: W_out = y * X^T * inv(X X^T + αI)
    Xt = states_w.T
    W_out = (y_w @ Xt) @ np.linalg.inv(Xt.T @ Xt + ridge_alpha * np.eye(n_reservoir))

    return W_in, W, W_out, x[:, [0]]  # return final state as last_state


def predict_esn(X, W_in, W, W_out, init_state, washout):
    n_inputs, T = X.shape
    x = init_state.copy()
    y_pred = np.zeros(T)
    for t in range(T):
        u = np.vstack([1, X[:, t:t+1]])
        x = np.tanh(W_in @ u + W @ x)
        if t >= washout:
            y_pred[t] = (W_out @ x)[0]
    return y_pred


# ------------------------------
# UI – Data Upload
# ------------------------------
st.sidebar.header("📂 Data Upload")

train_file = st.sidebar.file_uploader("Upload Training File (Excel)", type=['xlsx'])
test_file = st.sidebar.file_uploader("Upload Testing File (Excel)", type=['xlsx'])

if train_file:
    train_df = pd.read_excel(train_file)
    st.subheader("Training Data Preview")
    st.write(train_df.head())

    input_cols_train = st.multiselect("Select input columns for training", train_df.columns.tolist())
    target_col = st.selectbox("Select target column", train_df.columns.tolist())

    if test_file:
        test_df = pd.read_excel(test_file)
        st.subheader("Testing Data Preview")
        st.write(test_df.head())
        input_cols_test = st.multiselect("Select input columns for testing", test_df.columns.tolist(), default=input_cols_train)

        # ------------------------------
        # ESN Hyperparameters
        # ------------------------------
        st.sidebar.header("⚙️ ESN Hyperparameters")
        use_grid = st.sidebar.checkbox("Use Grid Search", value=False)

        if not use_grid:
            n_reservoir = st.sidebar.slider("Reservoir Size", 100, 2000, 500, 50)
            spectral_radius = st.sidebar.slider("Spectral Radius", 0.1, 2.0, 1.0, 0.1)
            ridge_alpha = st.sidebar.number_input("Ridge Regularization (α)", value=1e-6, format="%.1e")
            washout = st.sidebar.slider("Washout Steps", 0, 200, 50, 5)
        else:
            n_reservoir_list = st.sidebar.text_input("Reservoir Sizes (comma separated)", "200,500,800")
            spectral_list = st.sidebar.text_input("Spectral Radii (comma separated)", "0.5,0.8,1.0,1.2")
            ridge_list = st.sidebar.text_input("Ridge α (comma separated)", "1e-6,1e-5,1e-4")
            washout = st.sidebar.slider("Washout Steps", 0, 200, 50, 5)

            n_reservoir_list = [int(x.strip()) for x in n_reservoir_list.split(",")]
            spectral_list = [float(x.strip()) for x in spectral_list.split(",")]
            ridge_list = [float(x.strip()) for x in ridge_list.split(",")]

        # ------------------------------
        # Train button
        # ------------------------------
        if st.button("🚀 Train ESN"):
            # Scale data
            train_scaled, test_scaled, (y_min, y_max) = scale_data(train_df, test_df, input_cols_train, target_col)

            X_train = train_scaled[input_cols_train].to_numpy().T
            y_train = train_scaled[target_col].to_numpy()

            X_test = test_scaled[input_cols_test].to_numpy().T
            y_test_actual = test_df[target_col].to_numpy()

            if not use_grid:
                W_in, W, W_out, last_state = train_esn(X_train, y_train, n_reservoir, spectral_radius, ridge_alpha, washout)
                y_pred_scaled = predict_esn(X_test, W_in, W, W_out, last_state, washout)
                y_pred = inverse_scale(y_pred_scaled, y_min, y_max)

                # R2
                r2 = r2_score(y_test_actual[washout:], y_pred[washout:])
                st.success(f"✅ R² Score: {r2:.4f}")

            else:
                best_r2 = -np.inf
                best_params = None
                best_pred = None
                for nr in n_reservoir_list:
                    for sr in spectral_list:
                        for ra in ridge_list:
                            W_in, W, W_out, last_state = train_esn(X_train, y_train, nr, sr, ra, washout)
                            y_pred_scaled = predict_esn(X_test, W_in, W, W_out, last_state, washout)
                            y_pred = inverse_scale(y_pred_scaled, y_min, y_max)
                            r2 = r2_score(y_test_actual[washout:], y_pred[washout:])
                            if r2 > best_r2:
                                best_r2 = r2
                                best_params = (nr, sr, ra)
                                best_pred = y_pred
                st.success(f"🏆 Best R² = {best_r2:.4f} with params Reservoir={best_params[0]}, Spectral={best_params[1]}, Ridge={best_params[2]}")
                y_pred = best_pred

            # ------------------------------
            # Plotting
            # ------------------------------
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(y_test_actual, label="Actual")
            ax.plot(y_pred, label="Predicted", linestyle="--")
            ax.set_title("Predicted vs Actual")
            ax.legend()
            st.pyplot(fig)

            # ------------------------------
            # Download predictions
            # ------------------------------
            pred_df = pd.DataFrame({
                "Actual": y_test_actual,
                "Predicted": y_pred
            })
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                pred_df.to_excel(writer, index=False, sheet_name="Predictions")
            st.download_button(
                label="📥 Download Predictions",
                data=output.getvalue(),
                file_name="predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
