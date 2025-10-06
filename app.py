import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import eig
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from io import BytesIO
import itertools

# -----------------------------------
# PAGE CONFIG & TITLE
# -----------------------------------
st.set_page_config(page_title="ESN Predictor", layout="wide")
st.markdown("<h1 style='text-align:center;color:#2C3E50;'>⚡ Echo State Network Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------------
# FILE UPLOAD
# -----------------------------------
st.sidebar.header("📂 Data Upload")
train_file = st.sidebar.file_uploader("Upload Training Data (Excel or CSV)", type=["xlsx", "csv"])
test_file  = st.sidebar.file_uploader("Upload Test Data (Excel or CSV)", type=["xlsx", "csv"])

tabs = st.tabs(["🔧 Training & Grid Search", "📈 Prediction"])

# Store model
if 'trained_model' not in st.session_state:
    st.session_state['trained_model'] = None


# =========================================================
# 🔧 TRAINING TAB
# =========================================================
with tabs[0]:
    if train_file:
        # Load training data
        if train_file.name.endswith(".csv"):
            train_df = pd.read_csv(train_file)
        else:
            excel_sheets = pd.ExcelFile(train_file).sheet_names
            selected_sheet = st.selectbox("Select Sheet", excel_sheets)
            train_df = pd.read_excel(train_file, sheet_name=selected_sheet)

        st.subheader("👀 Training Data Preview")
        st.dataframe(train_df.head())

        input_cols = st.multiselect("Select Input Columns", train_df.columns.tolist(), default=train_df.columns[:-1].tolist())
        output_cols = st.multiselect("Select Output Column(s)", train_df.columns.tolist(), default=[train_df.columns[-1]])

        st.markdown("### ⚙️ ESN Hyperparameters")

        use_grid = st.checkbox("Use Grid Search to Find Best Hyperparameters")

        if use_grid:
            n_reservoir_range = st.text_input("Reservoir Sizes (comma-sep)", "50,100,200,300")
            sr_range = st.text_input("Spectral Radius (comma-sep)", "0.5,0.7,0.9,1.2")
            try:
                reservoir_options = [int(x.strip()) for x in n_reservoir_range.split(',')]
                sr_options = [float(x.strip()) for x in sr_range.split(',')]
            except:
                st.error("Invalid range input")
                st.stop()
        else:
            n_reservoir = st.slider("Reservoir Size", 10, 500, 100, 10)
            spectral_radius = st.slider("Spectral Radius", 0.1, 2.0, 0.9, 0.05)

        seed = st.number_input("Random Seed", value=0, step=1)
        np.random.seed(seed)

        # -------------------------------------------------
        # Training Button
        # -------------------------------------------------
        if st.button("🚀 Train ESN Model"):
            # Prepare data
            X = train_df[input_cols].values
            y = train_df[output_cols].values

            # Normalize
            X_mean, X_std = X.mean(axis=0), X.std(axis=0)
            y_mean, y_std = y.mean(axis=0), y.std(axis=0)

            Xn = ((X - X_mean) / X_std).T
            yn = ((y - y_mean) / y_std).T
            n_inputs = Xn.shape[0]

            # -----------------------------
            # ESN Training Function
            # -----------------------------
            def train_esn(n_res, sr, X_train, y_train):
                Win = (np.random.rand(n_res, n_inputs) * 2 - 1) * 0.1
                W = np.random.rand(n_res, n_res) * 2 - 1
                W *= sr / max(abs(eig(W)[0]))

                x = np.zeros((n_res, 1))
                states = []
                for t in range(X_train.shape[1]):
                    u = X_train[:, t:t+1]
                    x = np.tanh(Win @ u + W @ x)
                    states.append(x)
                states = np.hstack(states)

                reg = 1e-6
                Wout = y_train @ states.T @ np.linalg.pinv(states @ states.T + reg*np.eye(n_res))
                return Win, W, Wout

            def predict_esn(Win, W, Wout, X_data):
                n_res = W.shape[0]
                x = np.zeros((n_res,1))
                states = []
                for t in range(X_data.shape[1]):
                    u = X_data[:,t:t+1]
                    x = np.tanh(Win @ u + W @ x)
                    states.append(x)
                states = np.hstack(states)
                y_hat = Wout @ states
                return y_hat.T

            # -----------------------------
            # Grid Search with Validation
            # -----------------------------
            if use_grid:
                st.info("🔍 Running Grid Search (using validation split to improve test R²)")
                progress = st.progress(0)
                best_val_r2 = -np.inf
                best_model = None
                count = 0
                total = len(reservoir_options) * len(sr_options)

                # Split training set into train & validation
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

                Xtr_n = ((X_train - X_mean) / X_std).T
                ytr_n = ((y_train - y_mean) / y_std).T
                Xval_n = ((X_val - X_mean) / X_std).T
                yval = y_val

                for nr, sr in itertools.product(reservoir_options, sr_options):
                    count += 1
                    progress.progress(int(100 * count / total))

                    Win, W, Wout = train_esn(nr, sr, Xtr_n, ytr_n)
                    yval_pred_n = predict_esn(Win, W, Wout, Xval_n)
                    yval_pred = yval_pred_n * y_std + y_mean
                    r2_val = r2_score(yval, yval_pred)

                    if r2_val > best_val_r2:
                        best_val_r2 = r2_val
                        best_model = (Win, W, Wout, nr, sr)

                progress.empty()
                if best_model is None:
                    st.error("Grid Search failed to find good params.")
                else:
                    Win, W, Wout, n_reservoir, spectral_radius = best_model
                    st.success(f"✅ Best Validation R²: {best_val_r2:.4f} | Reservoir: {n_reservoir} | SR: {spectral_radius}")
            else:
                Win, W, Wout = train_esn(n_reservoir, spectral_radius, Xn, yn)
                yhat_train_n = predict_esn(Win, W, Wout, Xn)
                yhat_train = yhat_train_n * y_std + y_mean
                r2_train = r2_score(y, yhat_train)
                st.success(f"✅ Model Trained | Train R²: {r2_train:.4f}")

            # Store model
            st.session_state['trained_model'] = {
                "Win": Win,
                "W": W,
                "Wout": Wout,
                "X_mean": X_mean, "X_std": X_std,
                "y_mean": y_mean, "y_std": y_std,
                "input_cols": input_cols,
                "output_cols": output_cols,
                "n_reservoir": n_reservoir,
                "spectral_radius": spectral_radius
            }


# =========================================================
# 📈 PREDICTION TAB
# =========================================================
with tabs[1]:
    if test_file and st.session_state['trained_model'] is not None:
        # Load test data
        if test_file.name.endswith(".csv"):
            test_df = pd.read_csv(test_file)
        else:
            excel_sheets = pd.ExcelFile(test_file).sheet_names
            selected_sheet = st.selectbox("Select Sheet for Test Data", excel_sheets)
            test_df = pd.read_excel(test_file, sheet_name=selected_sheet)

        st.subheader("👀 Test Data Preview")
        st.dataframe(test_df.head())

        input_cols_test = st.multiselect("Select Input Columns for Prediction",
                                         test_df.columns.tolist(),
                                         default=st.session_state['trained_model']['input_cols'])

        if st.button("🔮 Predict"):
            model = st.session_state['trained_model']

            if set(input_cols_test) != set(model['input_cols']):
                st.error("Input columns for test do not match those used in training.")
                st.stop()

            Xtest = test_df[input_cols_test].values
            Xtest_n = ((Xtest - model['X_mean']) / model['X_std']).T

            # Predict
            def predict_esn(Win, W, Wout, X_data):
                n_res = W.shape[0]
                x = np.zeros((n_res,1))
                states = []
                for t in range(X_data.shape[1]):
                    u = X_data[:,t:t+1]
                    x = np.tanh(Win @ u + W @ x)
                    states.append(x)
                states = np.hstack(states)
                return (Wout @ states).T

            ypred_n = predict_esn(model['Win'], model['W'], model['Wout'], Xtest_n)
            ypred = ypred_n * model['y_std'] + model['y_mean']

            pred_df = pd.DataFrame(ypred, columns=model['output_cols'])
            st.subheader("📤 Predictions")
            st.dataframe(pred_df)

            # If actual outputs present
            if all(col in test_df.columns for col in model['output_cols']):
                ytrue = test_df[model['output_cols']].values
                r2_test = r2_score(ytrue, ypred)
                st.success(f"📈 Test R²: {r2_test:.4f}")

                fig, ax = plt.subplots(figsize=(10,5))
                ax.plot(ytrue, 'b', label="Actual", linewidth=2)
                ax.plot(ypred, 'r--', label="Predicted", linewidth=2)
                ax.legend(); ax.grid(); ax.set_title("Predicted vs Actual")
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots(figsize=(10,5))
                ax.plot(ypred, 'r--', label="Predicted")
                ax.legend(); ax.grid(); ax.set_title("Predicted Output")
                st.pyplot(fig)

            # Download predictions
            output = BytesIO()
            pred_df.to_excel(output, index=False, engine='openpyxl')
            st.download_button(
                "💾 Download Predictions",
                data=output.getvalue(),
                file_name="ESN_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
