

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

import torch


from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="Dự đoán giá xe ô tô", layout="wide")
st.title("🚗 Ứng dụng Dự đoán Giá xe Ô tô")

# Upload file
st.header("📤 Tải lên tập dữ liệu (.csv)")
uploaded_file = st.file_uploader("Chọn file CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Xem trước dữ liệu")
    st.dataframe(df.head())

    if 'price' not in df.columns:
        st.warning("❌ File CSV của bạn cần có cột 'price' để huấn luyện mô hình.")
    else:
        # Phân tích ban đầu
        st.subheader("🔍 Phân tích dữ liệu danh mục")

        # Plot: fuelType histogram
        st.markdown("#### 🔧 Phân phối 'fuelType'")
        fig1, ax1 = plt.subplots()
        ax1.hist(df['fuelType'], bins=len(df['fuelType'].unique()), edgecolor='black', linewidth=1.2)
        ax1.set_title("Fuel Types")
        st.pyplot(fig1)

        # Plot: transmission pie chart
        st.markdown("#### 🔧 Tỷ lệ 'transmission'")
        transmission_counts = df['transmission'].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(transmission_counts, autopct='%1.1f%%')
        ax2.legend(transmission_counts.index, loc="best")
        st.pyplot(fig2)

        # Plot: model histogram
        st.markdown("#### 🔧 Phân phối 'model'")
        fig3, ax3 = plt.subplots()
        ax3.hist(df['model'], bins=len(df['model'].unique()), edgecolor='black', linewidth=1.2)
        ax3.set_title("Model")
        st.pyplot(fig3)

        # Plot: manufacturer pie chart
        st.markdown("#### 🔧 Tỷ lệ 'Manufacturer'")
        manufacturer_counts = df['Manufacturer'].value_counts()
        fig4, ax4 = plt.subplots()
        ax4.pie(manufacturer_counts, autopct='%1.1f%%')
        ax4.legend(manufacturer_counts.index, loc="best")
        st.pyplot(fig4)

        # Thêm checkbox gộp fuelType
        st.markdown("#### 🛠️ Tùy chọn gộp fuelType")
        if st.checkbox("Gộp 'Electric' và 'Other' thành 'Other' trong fuelType", value=True):
            df['fuelType'] = df['fuelType'].replace(['Electric', 'Other'], 'Other')

        # One-hot encoding
        df = pd.get_dummies(df, drop_first=True)

        # Checkbox gộp Manufacturer nhỏ
        st.markdown("#### 🛠️ Tùy chọn gộp các hãng nhỏ")
        if st.checkbox("Gộp các hãng nhỏ lại thành Manufacturer_Other", value=True):
            manuf_cols = ['Manufacturer_skoda', 'Manufacturer_toyota', 'Manufacturer_vauxhall',
                          'Manufacturer_volkswagen', 'Manufacturer_hyundi']
            if all(col in df.columns for col in manuf_cols):
                df['Manufacturer_Other'] = (
                    df['Manufacturer_skoda'] +
                    df['Manufacturer_toyota'] +
                    df['Manufacturer_vauxhall'] +
                    df['Manufacturer_volkswagen'] +
                    df['Manufacturer_hyundi']
                ).clip(upper=1)
                df.drop(columns=manuf_cols, inplace=True)

        # Hiển thị phân phối giá và boxplot
        st.subheader("📉 Biểu đồ phân phối và boxplot giá xe")
        # ... (histplot và boxplot như cũ)

        # Scatter
        st.markdown("## 📊 Scatter: Giá xe và các biến số khác")
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if 'price' in numeric_cols:
            numeric_cols.remove('price')
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=col, y='price', alpha=0.6, ax=ax)
            ax.set_title(f"Giá xe vs {col}")
            st.pyplot(fig)

        # Bắt đầu huấn luyện mô hình
        st.header("⚙️ Cấu hình mô hình")
        target = 'price'
        X = df.drop(columns=[target])
        y = df[target]

        st.subheader("🔍 Gợi ý chuẩn hóa từng cột")
        scaler_choices = {}
        X_scaled_df = pd.DataFrame()

        for col in X.columns:
            # Chuyển dữ liệu về kiểu float để tránh lỗi với kiểu bool
            col_data = X[[col]].astype(float)

            q05, q95 = col_data[col].quantile(0.05), col_data[col].quantile(0.95)
            outlier_ratio = ((col_data[col] < q05) | (col_data[col] > q95)).sum() / len(col_data)
            suggested = 'RobustScaler' if outlier_ratio > 0.05 else 'MinMaxScaler'
    
            choice = st.selectbox(f"{col} (outliers ≈ {outlier_ratio:.1%})", ['MinMaxScaler', 'RobustScaler'],
                          index=1 if suggested == 'RobustScaler' else 0)
    
            scaler = RobustScaler() if choice == 'RobustScaler' else MinMaxScaler()
            X_scaled_df[col] = scaler.fit_transform(col_data).flatten()
            scaler_choices[col] = choice

        st.subheader(" Chọn mô hình dự đoán")
        model_choice = st.selectbox("Chọn mô hình huấn luyện", ['Linear Regression', 'Decision Tree', 'Tuned Decision Tree',
        'Random Forest','Tuned Random Forest','Gradient Boosting','Tuned Gradient Boosting','XGBoost'])

        if model_choice == 'Linear Regression':
            model = LinearRegression()
        elif model_choice == 'Decision Tree':
            model = DecisionTreeRegressor(random_state=42)
        elif model_choice == 'Tuned Decision Tree':
            model = DecisionTreeRegressor(max_depth= None, min_samples_leaf = 2, min_samples_split = 10, random_state=42)
        elif model_choice == 'Random Forest':
            model = RandomForestRegressor(random_state=42)
        elif model_choice == 'Tuned Random Forest':
            model = RandomForestRegressor(max_depth = 20, min_samples_split = 3, n_estimators = 76 , random_state=42)
        elif model_choice == 'Gradient Boosting':
            model = GradientBoostingRegressor(random_state=42)
        elif model_choice == 'Tuned Gradient Boosting':
            model = GradientBoostingRegressor(learning_rate= 0.06876831063110332, max_depth = 9, min_samples_split = 3, n_estimators =98,random_state=42)
        elif model_choice == 'XGBoost':
            model = XGBRegressor(random_state=42, verbosity=0,colsample_bytree=0.705051979426657, learning_rate= 0.11883357504897696, max_depth=10, n_estimators = 66, subsample=0.8350739741344673)
        else:
            model = TabNetRegressor(seed=42, verbose=0)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
        if model_choice == 'TabNet':
           model.fit(
            X_train.values.astype('float32'),
            y_train.values.reshape(-1, 1).astype('float32'),
            eval_set=[(
            X_test.values.astype('float32'),
            y_test.values.reshape(-1, 1).astype('float32')
            )],
            patience=20,
            max_epochs=200,
            batch_size=1024,
            virtual_batch_size=128
            )
           y_pred = model.predict(X_test.values.astype('float32')).flatten()
        else:
           model.fit(X_train, y_train)
           y_pred = model.predict(X_test)

        st.success("✅ Mô hình đã huấn luyện xong!")

        st.subheader("📈 Đánh giá mô hình")

        # Dự đoán và đánh giá
        y_pred = model.predict(X_test)

        # Các chỉ số
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # MAPE chỉ tính với giá trị thật khác 0
        non_zero = y_test != 0
        mape = np.mean(np.abs((y_test[non_zero] - y_pred[non_zero]) / y_test[non_zero])) * 100

        # Hiển thị
        st.write(f"**MAE (Mean Absolute Error)**: {mae:,.0f}")
        st.write(f"**MSE (Mean Squared Error)**: {mse:,.0f}")
        st.write(f"**RMSE (Root Mean Squared Error)**: {rmse:,.0f}")
        st.write(f"**R² Score**: {r2:.3f}")
        st.write(f"**MAPE (Mean Absolute Percentage Error)**: {mape:.2f}%")

        # Biểu đồ scatter: Thực tế vs Dự đoán với 2 màu khác nhau
        fig7, ax7 = plt.subplots()
        ax7.scatter(range(len(y_test)), y_test, label='Giá thực tế', color='blue', alpha=0.6)
        ax7.scatter(range(len(y_pred)), y_pred, label='Giá dự đoán', color='red', alpha=0.6)
        ax7.set_xlabel("Index mẫu kiểm tra")
        ax7.set_ylabel("Giá xe")
        ax7.set_title("So sánh Giá Thực tế và Giá Dự đoán")
        ax7.legend()
        st.pyplot(fig7)

