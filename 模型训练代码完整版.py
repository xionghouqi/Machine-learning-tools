import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import joblib
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten, MaxPooling1D, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_score
from mlxtend.regressor import StackingRegressor
import openpyxl
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
import base64
from io import BytesIO
import os
from datetime import datetime
import json  # 用于导出JSON格式的图表数据

# 尝试导入statsmodels，如果失败则给出警告
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# 设置页面布局
st.set_page_config(layout="wide", page_title="机器学习模型集成分析系统")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 标题
st.title("机器学习模型集成与可视化分析系统")

# 语言选择
language = st.sidebar.selectbox("选择语言/Select Language", ["中文", "English"])

# 翻译字典
translations = {
    "中文": {
        "upload_data": "上传数据集 (优先Excel, 也支持CSV)",
        "show_data": "显示完整数据",
        "data_preprocessing": "数据预处理",
        "missing_value": "处理缺失值方法",
        "missing_options": ["删除缺失行", "均值填充", "中位数填充", "众数填充"],
        "scaling": "特征缩放方法",
        "scaling_options": ["无", "标准化 (StandardScaler)", "归一化 (MinMaxScaler)"],
        "data_split": "数据划分",
        "test_size": "测试集比例",
        "random_state": "随机种子",
        "target": "选择目标变量",
        "features": "选择特征变量",
        "data_exploration": "数据探索性分析",
        "stats": "数据统计信息",
        "correlation": "特征相关性分析",
        "distribution": "特征分布",
        "target_dist": "目标变量分布",
        "3d_plot": "3D特征关系图",
        "model_training": "模型训练",
        "select_models": "选择要训练的模型",
        "model_options": [
            "线性回归", "支持向量回归", "随机森林", "XGBoost", 
            "LightGBM", "高斯过程", "Stacking集成", 
            "贝叶斯岭回归", "多层感知机", "LSTM网络",
            "决策树", "K近邻", "卷积神经网络", "循环神经网络"
        ],
        "start_training": "开始训练",
        "model_evaluation": "模型评估",
        "performance": "模型性能比较",
        "detailed_analysis": "模型详细分析",
        "visualization": "可视化分析",
        "model_saving": "模型保存与导出",
        "save_model": "保存模型",
        "export_results": "导出结果",
        "export_charts": "导出图表",
        "time_series": "时间序列分析",
        "export_all_charts": "导出所有图表",
        "export_format": "导出格式",
        "export_success": "图表已成功导出到 charts 文件夹!"
    },
    "English": {
        "upload_data": "Upload Dataset (Excel preferred, CSV also supported)",
        "show_data": "Show Full Data",
        "data_preprocessing": "Data Preprocessing",
        "missing_value": "Missing Value Handling",
        "missing_options": ["Drop missing rows", "Mean imputation", "Median imputation", "Mode imputation"],
        "scaling": "Feature Scaling Method",
        "scaling_options": ["None", "Standardization (StandardScaler)", "Normalization (MinMaxScaler)"],
        "data_split": "Data Splitting",
        "test_size": "Test Size Ratio",
        "random_state": "Random State",
        "target": "Select Target Variable",
        "features": "Select Feature Variables",
        "data_exploration": "Data Exploration",
        "stats": "Data Statistics",
        "correlation": "Feature Correlation Analysis",
        "distribution": "Feature Distribution",
        "target_dist": "Target Variable Distribution",
        "3d_plot": "3D Feature Relationship",
        "model_training": "Model Training",
        "select_models": "Select Models to Train",
        "model_options": [
            "Linear Regression", "Support Vector Regression", "Random Forest", "XGBoost", 
            "LightGBM", "Gaussian Process", "Stacking Ensemble", 
            "Bayesian Ridge", "MLP", "LSTM Network",
            "Decision Tree", "K-Nearest Neighbors", "CNN", "RNN"
        ],
        "start_training": "Start Training",
        "model_evaluation": "Model Evaluation",
        "performance": "Model Performance Comparison",
        "detailed_analysis": "Detailed Model Analysis",
        "visualization": "Visualization Analysis",
        "model_saving": "Model Saving & Export",
        "save_model": "Save Model",
        "export_results": "Export Results",
        "export_charts": "Export Charts",
        "time_series": "Time Series Analysis",
        "export_all_charts": "Export All Charts",
        "export_format": "Export Format",
        "export_success": "Charts successfully exported to charts folder!"
    }
}

def t(key):
    """翻译函数"""
    return translations[language][key]

# 添加save_chart函数
def save_chart(fig, filename, data=None, folder="charts", format="png", dpi=300):
    """
    保存图表到文件和对应的数据文件
    返回包含成功状态和文件路径的字典
    """
    # 确保目录存在
    os.makedirs(folder, exist_ok=True)
    result = {"success": False, "format": None, "filepath": None, "data_filepath": None}
    
    # 保存图表数据为CSV (用于Origin绘图)
    if data is not None:
        try:
            data_path = f"{folder}/{filename}_data.csv"
            if isinstance(data, pd.DataFrame):
                data.to_csv(data_path, index=False)
            elif isinstance(data, dict):
                pd.DataFrame(data).to_csv(data_path, index=False)
            result["data_filepath"] = data_path
        except Exception as e:
            st.warning(f"保存图表数据失败: {str(e)}" if language == "中文" else f"Failed to save chart data: {str(e)}")
    
    # 保存图表图像
    try:
        if isinstance(fig, plt.Figure):
            # Matplotlib图表
            if format == "html":
                # 如果需要HTML格式，保存为PNG后转HTML
                filepath = os.path.join(folder, f"{filename}.png")
                fig.savefig(filepath, format="png", dpi=dpi, bbox_inches="tight")
                
                # 创建一个包含图像的简单HTML文件
                html_path = os.path.join(folder, f"{filename}.html")
                with open(html_path, "w") as f:
                    f.write(f"<html><body><img src='{os.path.basename(filepath)}' /></body></html>")
                
                result["success"] = True
                result["format"] = "html"
                result["filepath"] = html_path
            else:
                # 直接保存为要求的格式
                filepath = f"{folder}/{filename}.{format}"
                fig.savefig(filepath, format=format, dpi=dpi, bbox_inches="tight")
                result["success"] = True
                result["format"] = format
                result["filepath"] = filepath
        else:
            # Plotly图表 - 直接保存为HTML，忽略其他格式
            html_path = f"{folder}/{filename}.html"
            fig.write_html(html_path)
            result["success"] = True
            result["format"] = "html"
            result["filepath"] = html_path
            
            # 如果用户原本想要其他格式，给出提示
            if format != "html":
                st.info(f"Plotly图表已保存为HTML格式。如需其他格式，请安装kaleido: pip install kaleido" if language == "中文" 
                      else f"Plotly chart saved as HTML. For other formats, install kaleido: pip install kaleido")
    except Exception as e:
        st.error(f"导出图表失败: {str(e)}" if language == "中文" else f"Failed to export chart: {str(e)}")
    
    return result

# 初始化session state
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'language' not in st.session_state:
    st.session_state.language = language

# 侧边栏 - 数据上传和预处理
with st.sidebar:
    st.header(t("data_preprocessing"))
    
    # 上传数据
    uploaded_file = st.file_uploader(t("upload_data"), type=["xlsx", "xls", "csv"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(uploaded_file)
            else:
                data = pd.read_csv(uploaded_file)
            
            st.session_state.data = data
            st.success("数据上传成功!" if language == "中文" else "Data uploaded successfully!")
            
            # 显示完整数据
            if st.checkbox(t("show_data")):
                st.dataframe(data)
                
            # 数据预处理选项
            st.subheader(t("data_preprocessing"))
            
            # 处理缺失值
            if data.isnull().sum().sum() > 0:
                st.warning("数据中存在缺失值!" if language == "中文" else "Missing values detected!")
                missing_option = st.selectbox(
                    t("missing_value"),
                    t("missing_options")
                )
                
                if missing_option == t("missing_options")[0]:
                    data = data.dropna()
                elif missing_option == t("missing_options")[1]:
                    data = data.fillna(data.mean())
                elif missing_option == t("missing_options")[2]:
                    data = data.fillna(data.median())
                else:
                    for col in data.columns:
                        if data[col].dtype == 'object':
                            data[col] = data.fillna(data[col].mode()[0])
            
            # 特征缩放
            scale_option = st.selectbox(
                t("scaling"),
                t("scaling_options")
            )
            
            # 数据分割
            st.subheader(t("data_split"))
            test_size = st.slider(t("test_size"), 0.1, 0.5, 0.2, 0.05, key="test_size_slider")
            random_state = st.number_input(t("random_state"), 0, 100, 42, key="random_state_input")
            
            # 选择特征和目标
            target_col = st.selectbox(t("target"), data.columns, key="target_select")
            feature_cols = st.multiselect(t("features"), [col for col in data.columns if col != target_col], key="features_select")
            
            if feature_cols and target_col:
                X = data[feature_cols]
                y = data[target_col]
                
                # 应用特征缩放
                if scale_option == t("scaling_options")[1]:
                    scaler = StandardScaler()
                    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
                elif scale_option == t("scaling_options")[2]:
                    scaler = MinMaxScaler()
                    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
                
                # 分割数据集
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                st.success(f"{'数据划分完成!' if language == '中文' else 'Data split completed!'} 训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")
                
                # 保存到session state
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.feature_cols = feature_cols
                st.session_state.target_col = target_col
                st.session_state.scaler = scaler if scale_option != t("scaling_options")[0] else None
        
        except Exception as e:
            st.error(f"{'数据读取错误:' if language == '中文' else 'Error reading data:'} {str(e)}")

# 主界面
if 'X_train' in st.session_state:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        t("data_exploration"), t("model_training"), t("model_evaluation"), 
        t("visualization"), t("model_saving")
    ])
    
    with tab1:
        st.header(t("data_exploration"))
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 数据统计信息
            st.subheader(t("stats"))
            st.dataframe(st.session_state.X_train.describe())
            
            # 相关性分析
            st.subheader(t("correlation"))
            corr_matrix = pd.concat([st.session_state.X_train, st.session_state.y_train], axis=1).corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
            
        with col2:
            # 特征分布
            st.subheader(t("distribution"))
            selected_feature = st.selectbox(t("distribution"), st.session_state.feature_cols, key="feature_dist_select")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(st.session_state.X_train[selected_feature], kde=True, ax=ax)
            ax.set_xlabel(selected_feature)
            ax.set_ylabel('Frequency' if language == "English" else '频率')
            st.pyplot(fig)
            
            # 目标变量分布
            st.subheader(t("target_dist"))
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(st.session_state.y_train, kde=True, ax=ax)
            ax.set_xlabel(st.session_state.target_col)
            ax.set_ylabel('Frequency' if language == "English" else '频率')
            st.pyplot(fig)
            
            # 3D散点图（如果特征数量>=2）
            if len(st.session_state.feature_cols) >= 2:
                st.subheader(t("3d_plot"))
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                x_feature = st.selectbox("X轴特征" if language == "中文" else "X Feature", 
                                       st.session_state.feature_cols, index=0, key="x_feature_select")
                y_feature = st.selectbox("Y轴特征" if language == "中文" else "Y Feature", 
                                       st.session_state.feature_cols, index=1, key="y_feature_select")
                
                if len(st.session_state.feature_cols) > 2:
                    z_feature = st.selectbox("Z轴特征" if language == "中文" else "Z Feature", 
                                           st.session_state.feature_cols, index=2, key="z_feature_select")
                else:
                    z_feature = y_feature
                
                ax.scatter(
                    st.session_state.X_train[x_feature],
                    st.session_state.X_train[y_feature],
                    st.session_state.X_train[z_feature],
                    c=st.session_state.y_train,
                    cmap='viridis'
                )
                ax.set_xlabel(x_feature)
                ax.set_ylabel(y_feature)
                ax.set_zlabel(z_feature)
                st.pyplot(fig)
    
    with tab2:
        st.header(t("model_training"))
        
        model_options = st.multiselect(
            t("select_models"),
            t("model_options"),
            default=[t("model_options")[0], t("model_options")[2]],
            key="model_select"
        )
        
        # 模型参数设置
        model_params = {}
        
        if t("model_options")[0] in model_options:  # 线性回归
            with st.expander("线性回归参数" if language == "中文" else "Linear Regression Parameters"):
                model_params["linear_regression"] = {
                    "fit_intercept": st.checkbox("拟合截距" if language == "中文" else "Fit intercept", True, key="linear_fit_intercept")
                }
        
        if t("model_options")[1] in model_options:  # 支持向量回归
            with st.expander("支持向量回归参数" if language == "中文" else "SVR Parameters"):
                model_params["svr"] = {
                    "kernel": st.selectbox("核函数" if language == "中文" else "Kernel", 
                                         ["rbf", "linear", "poly", "sigmoid"], index=0, key="svr_kernel"),
                    "C": st.slider("C (正则化参数)" if language == "中文" else "C (Regularization)", 
                                 0.1, 10.0, 1.0, 0.1, key="svr_c"),
                    "epsilon": st.slider("ε (epsilon)" if language == "中文" else "ε (epsilon)", 
                                      0.01, 1.0, 0.1, 0.01, key="svr_epsilon")
                }
        
        if t("model_options")[2] in model_options:  # 随机森林
            with st.expander("随机森林参数" if language == "中文" else "Random Forest Parameters"):
                model_params["random_forest"] = {
                    "n_estimators": st.slider("树的数量" if language == "中文" else "Number of trees", 
                                            10, 500, 100, 10, key="rf_n_estimators"),
                    "max_depth": st.selectbox("最大深度" if language == "中文" else "Max depth", 
                                           [None, 5, 10, 20, 30], index=0, key="rf_max_depth"),
                    "min_samples_split": st.slider("最小分裂样本数" if language == "中文" else "Min samples split", 
                                                2, 20, 2, 1, key="rf_min_samples_split")
                }
        
        if t("model_options")[3] in model_options:  # XGBoost
            with st.expander("XGBoost参数" if language == "中文" else "XGBoost Parameters"):
                model_params["xgboost"] = {
                    "n_estimators": st.slider("树的数量" if language == "中文" else "Number of trees", 
                                            10, 500, 100, 10, key="xgb_n_estimators"),
                    "max_depth": st.slider("最大深度" if language == "中文" else "Max depth", 
                                        1, 15, 6, 1, key="xgb_max_depth"),
                    "learning_rate": st.slider("学习率" if language == "中文" else "Learning rate", 
                                            0.01, 1.0, 0.3, 0.01, key="xgb_learning_rate")
                }
        
        if t("model_options")[4] in model_options:  # LightGBM
            with st.expander("LightGBM参数" if language == "中文" else "LightGBM Parameters"):
                model_params["lightgbm"] = {
                    "n_estimators": st.slider("树的数量" if language == "中文" else "Number of trees", 
                                            10, 500, 100, 10, key="lgbm_n_estimators"),
                    "max_depth": st.slider("最大深度" if language == "中文" else "Max depth", 
                                        1, 15, -1, 1, key="lgbm_max_depth"),
                    "learning_rate": st.slider("学习率" if language == "中文" else "Learning rate", 
                                            0.01, 1.0, 0.1, 0.01, key="lgbm_learning_rate")
                }
        
        if t("model_options")[5] in model_options:  # 高斯过程
            with st.expander("高斯过程参数" if language == "中文" else "Gaussian Process Parameters"):
                model_params["gaussian_process"] = {
                    "alpha": st.slider("α (alpha)" if language == "中文" else "α (alpha)", 
                                    0.0001, 1.0, 0.1, 0.0001, key="gp_alpha")
                }
        
        if t("model_options")[6] in model_options:  # Stacking集成
            with st.expander("Stacking集成参数" if language == "中文" else "Stacking Parameters"):
                model_params["stacking"] = {
                    "meta_model": st.selectbox("元模型" if language == "中文" else "Meta model", 
                                            ["LinearRegression", "RandomForest"], index=0, key="stacking_meta_model")
                }
        
        if t("model_options")[7] in model_options:  # 贝叶斯岭回归
            with st.expander("贝叶斯岭回归参数" if language == "中文" else "Bayesian Ridge Parameters"):
                model_params["bayesian_ridge"] = {
                    "max_iter": st.slider("最大迭代次数" if language == "中文" else "Maximum iterations", 
                                     100, 1000, 300, 50, key="br_max_iter")
                }
        
        if t("model_options")[8] in model_options:  # 多层感知机
            with st.expander("多层感知机参数" if language == "中文" else "MLP Parameters"):
                model_params["mlp"] = {
                    "hidden_layer_sizes": st.text_input("隐藏层结构" if language == "中文" else "Hidden layer sizes", 
                                                     "100,50", key="mlp_hidden_layers"),
                    "activation": st.selectbox("激活函数" if language == "中文" else "Activation", 
                                            ["relu", "tanh", "logistic"], index=0, key="mlp_activation"),
                    "solver": st.selectbox("优化器" if language == "中文" else "Solver", 
                                         ["adam", "sgd", "lbfgs"], index=0, key="mlp_solver"),
                    "max_iter": st.slider("最大迭代次数" if language == "中文" else "Max iterations", 
                                       100, 2000, 500, 50, key="mlp_max_iter")
                }
        
        if t("model_options")[9] in model_options:  # LSTM网络
            with st.expander("LSTM网络参数" if language == "中文" else "LSTM Parameters"):
                model_params["lstm"] = {
                    "units": st.slider("LSTM单元数" if language == "中文" else "LSTM units", 
                                    10, 200, 50, 10, key="lstm_units"),
                    "epochs": st.slider("训练轮次" if language == "中文" else "Epochs", 
                                     10, 500, 100, 10, key="lstm_epochs"),
                    "batch_size": st.slider("批量大小" if language == "中文" else "Batch size", 
                                         16, 256, 32, 16, key="lstm_batch_size")
                }
        
        if t("model_options")[10] in model_options:  # 决策树
            with st.expander("决策树参数" if language == "中文" else "Decision Tree Parameters"):
                model_params["decision_tree"] = {
                    "max_depth": st.slider("最大深度" if language == "中文" else "Max depth", 
                                         None, 30, 10, 1, key="dt_max_depth"),
                    "min_samples_split": st.slider("最小分裂样本数" if language == "中文" else "Min samples split", 
                                                2, 20, 2, 1, key="dt_min_samples_split"),
                    "criterion": st.selectbox("分裂标准" if language == "中文" else "Criterion", 
                                           ["squared_error", "friedman_mse", "absolute_error", "poisson"], 
                                           index=0, key="dt_criterion")
                }
        
        if t("model_options")[11] in model_options:  # K近邻
            with st.expander("K近邻参数" if language == "中文" else "KNN Parameters"):
                model_params["knn"] = {
                    "n_neighbors": st.slider("邻居数量" if language == "中文" else "Number of neighbors", 
                                          1, 20, 5, 1, key="knn_n_neighbors"),
                    "weights": st.selectbox("权重" if language == "中文" else "Weights", 
                                         ["uniform", "distance"], index=0, key="knn_weights"),
                    "algorithm": st.selectbox("算法" if language == "中文" else "Algorithm", 
                                           ["auto", "ball_tree", "kd_tree", "brute"], index=0, key="knn_algorithm")
                }
        
        if t("model_options")[12] in model_options:  # 卷积神经网络
            with st.expander("卷积神经网络参数" if language == "中文" else "CNN Parameters"):
                model_params["cnn"] = {
                    "filters": st.slider("卷积核数量" if language == "中文" else "Number of filters", 
                                      16, 128, 64, 16, key="cnn_filters"),
                    "kernel_size": st.slider("卷积核大小" if language == "中文" else "Kernel size", 
                                          2, 5, 3, 1, key="cnn_kernel_size"),
                    "epochs": st.slider("训练轮次" if language == "中文" else "Epochs", 
                                     10, 500, 100, 10, key="cnn_epochs"),
                    "batch_size": st.slider("批量大小" if language == "中文" else "Batch size", 
                                         16, 256, 32, 16, key="cnn_batch_size")
                }
        
        if t("model_options")[13] in model_options:  # 循环神经网络
            with st.expander("循环神经网络参数" if language == "中文" else "RNN Parameters"):
                model_params["rnn"] = {
                    "units": st.slider("RNN单元数" if language == "中文" else "RNN units", 
                                    10, 200, 50, 10, key="rnn_units"),
                    "epochs": st.slider("训练轮次" if language == "中文" else "Epochs", 
                                     10, 500, 100, 10, key="rnn_epochs"),
                    "batch_size": st.slider("批量大小" if language == "中文" else "Batch size", 
                                         16, 256, 32, 16, key="rnn_batch_size")
                }
        
        if st.button(t("start_training"), key="train_button"):
            if not model_options:
                st.warning("请至少选择一个模型!" if language == "中文" else "Please select at least one model!")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                X_train = st.session_state.X_train.values
                X_test = st.session_state.X_test.values
                y_train = st.session_state.y_train.values
                y_test = st.session_state.y_test.values
                
                # 准备LSTM数据
                def prepare_lstm_data(X, y, time_steps=1):
                    Xs, ys = [], []
                    for i in range(len(X) - time_steps):
                        v = X[i:(i + time_steps)]
                        Xs.append(v)
                        ys.append(y[i + time_steps])
                    return np.array(Xs), np.array(ys)
                
                time_steps = 3
                X_train_lstm, y_train_lstm = prepare_lstm_data(X_train, y_train, time_steps)
                X_test_lstm, y_test_lstm = prepare_lstm_data(X_test, y_test, time_steps)
                
                models = {}
                results = {}
                
                for i, model_name in enumerate(model_options):
                    progress = (i + 1) / len(model_options)
                    progress_bar.progress(progress)
                    status_text.text(f"{'正在训练' if language == '中文' else 'Training'} {model_name}...")
                    
                    start_time = time.time()
                    
                    if model_name == t("model_options")[0]:  # 线性回归
                        params = model_params.get("linear_regression", {})
                        model = LinearRegression(**params)
                        model.fit(X_train, y_train)
                        
                    elif model_name == t("model_options")[1]:  # 支持向量回归
                        params = model_params.get("svr", {})
                        model = SVR(**params)
                        model.fit(X_train, y_train)
                        
                    elif model_name == t("model_options")[2]:  # 随机森林
                        params = model_params.get("random_forest", {})
                        model = RandomForestRegressor(**params, random_state=random_state)
                        model.fit(X_train, y_train)
                        
                    elif model_name == t("model_options")[3]:  # XGBoost
                        params = model_params.get("xgboost", {})
                        model = XGBRegressor(**params, random_state=random_state)
                        model.fit(X_train, y_train)
                        
                    elif model_name == t("model_options")[4]:  # LightGBM
                        params = model_params.get("lightgbm", {})
                        model = LGBMRegressor(**params, random_state=random_state)
                        model.fit(X_train, y_train)
                        
                    elif model_name == t("model_options")[5]:  # 高斯过程
                        params = model_params.get("gaussian_process", {})
                        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
                        model = GaussianProcessRegressor(kernel=kernel, **params)
                        model.fit(X_train, y_train)
                        
                    elif model_name == t("model_options")[6]:  # Stacking集成
                        params = model_params.get("stacking", {})
                        # 创建基础回归器列表（不是元组）
                        base_models = [
                            LinearRegression(),
                            SVR(),
                            RandomForestRegressor(n_estimators=100)
                        ]
                        # 创建元模型
                        meta_model = LinearRegression() if params["meta_model"] == "LinearRegression" else RandomForestRegressor()
                        # 创建Stacking回归器
                        model = StackingRegressor(
                            regressors=base_models,
                            meta_regressor=meta_model
                        )
                        model.fit(X_train, y_train)
                        
                    elif model_name == t("model_options")[7]:  # 贝叶斯岭回归
                        # 获取参数并确保参数有效
                        params = model_params.get("bayesian_ridge", {})
                        
                        # 将n_iter参数改为max_iter
                        if "max_iter" in params:
                            params["max_iter"] = int(params["max_iter"])
                        
                        try:
                            # 尝试创建模型并查看是否有参数问题
                            model = BayesianRidge(**params)
                            model.fit(X_train, y_train)
                        except Exception as e:
                            # 如果出现错误，使用默认参数
                            st.error(f"贝叶斯岭回归参数错误: {str(e)}" if language == "中文" else f"Error with Bayesian Ridge parameters: {str(e)}")
                            st.warning("使用默认参数" if language == "中文" else "Using default parameters")
                            model = BayesianRidge()
                            model.fit(X_train, y_train)
                        
                    elif model_name == t("model_options")[8]:  # 多层感知机
                        params = model_params.get("mlp", {})
                        hidden_layers = tuple(map(int, params["hidden_layer_sizes"].split(',')))
                        model = MLPRegressor(
                            hidden_layer_sizes=hidden_layers,
                            activation=params["activation"],
                            solver=params["solver"],
                            max_iter=params["max_iter"],
                            random_state=random_state
                        )
                        model.fit(X_train, y_train)
                        
                    elif model_name == t("model_options")[9]:  # LSTM网络
                        params = model_params.get("lstm", {})
                        model = Sequential([
                            LSTM(params["units"], activation='relu', input_shape=(time_steps, X_train.shape[1])),
                            Dense(1)
                        ])
                        model.compile(optimizer='adam', loss='mse')
                        history = model.fit(
                            X_train_lstm, y_train_lstm,
                            epochs=params["epochs"],
                            batch_size=params["batch_size"],
                            validation_split=0.2,
                            callbacks=[EarlyStopping(patience=10)],
                            verbose=0
                        )
                        # 保存训练历史
                        results[model_name + "_history"] = history.history
                    
                    elif model_name == t("model_options")[10]:  # 决策树
                        params = model_params.get("decision_tree", {})
                        model = DecisionTreeRegressor(**params, random_state=random_state)
                        model.fit(X_train, y_train)
                    
                    elif model_name == t("model_options")[11]:  # K近邻
                        params = model_params.get("knn", {})
                        model = KNeighborsRegressor(**params)
                        model.fit(X_train, y_train)
                    
                    elif model_name == t("model_options")[12]:  # 卷积神经网络
                        params = model_params.get("cnn", {})
                        # 准备CNN数据
                        X_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                        X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                        
                        model = Sequential([
                            Conv1D(filters=params["filters"], kernel_size=params["kernel_size"], activation='relu', input_shape=(X_train.shape[1], 1)),
                            MaxPooling1D(2),
                            Flatten(),
                            Dense(64, activation='relu'),
                            Dense(1)
                        ])
                        model.compile(optimizer='adam', loss='mse')
                        history = model.fit(
                            X_cnn, y_train,
                            epochs=params["epochs"],
                            batch_size=params["batch_size"],
                            validation_split=0.2,
                            callbacks=[EarlyStopping(patience=10)],
                            verbose=0
                        )
                        results[model_name + "_history"] = history.history
                    
                    elif model_name == t("model_options")[13]:  # 循环神经网络
                        params = model_params.get("rnn", {})
                        # 准备RNN数据
                        X_rnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                        X_test_rnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                        
                        model = Sequential([
                            SimpleRNN(params["units"], activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=False),
                            Dense(1)
                        ])
                        model.compile(optimizer='adam', loss='mse')
                        history = model.fit(
                            X_rnn, y_train,
                            epochs=params["epochs"],
                            batch_size=params["batch_size"],
                            validation_split=0.2,
                            callbacks=[EarlyStopping(patience=10)],
                            verbose=0
                        )
                        results[model_name + "_history"] = history.history
                    
                    # 预测和评估
                    if model_name in [t("model_options")[9], t("model_options")[12], t("model_options")[13]]:  # LSTM, CNN, RNN
                        if model_name == t("model_options")[9]:  # LSTM
                            y_pred = model.predict(X_test_lstm).flatten()
                            y_true = y_test[time_steps:]
                        elif model_name == t("model_options")[12]:  # CNN
                            y_pred = model.predict(X_test_cnn).flatten()
                            y_true = y_test
                        elif model_name == t("model_options")[13]:  # RNN
                            y_pred = model.predict(X_test_rnn).flatten()
                            y_true = y_test
                    else:
                        y_pred = model.predict(X_test)
                        y_true = y_test
                    
                    mse = mean_squared_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)
                    mae = mean_absolute_error(y_true, y_pred)
                    
                    training_time = time.time() - start_time
                    
                    models[model_name] = model
                    results[model_name] = {
                        'mse': mse,
                        'r2': r2,
                        'mae': mae,
                        'training_time': training_time,
                        'y_pred': y_pred,
                        'y_true': y_true
                    }
                
                st.session_state.models = models
                st.session_state.results = results
                progress_bar.empty()
                status_text.text("所有模型训练完成!" if language == "中文" else "All models trained successfully!")
                
    with tab3:
        st.header(t("model_evaluation"))
        
        if not st.session_state.models:
            st.warning("请先训练模型!" if language == "中文" else "Please train models first!")
        else:
            # 模型性能比较
            st.subheader(t("performance"))
            
            metrics_df = pd.DataFrame(columns=[
                'Model', 'MSE', 'R2 Score', 'MAE', 'Training Time (s)'
            ])
            
            for model_name, result in st.session_state.results.items():
                if "_history" not in model_name:  # 排除LSTM的训练历史
                    metrics_df = pd.concat([metrics_df, pd.DataFrame({
                        'Model': [model_name],
                        'MSE': [result['mse']],
                        'R2 Score': [result['r2']],
                        'MAE': [result['mae']],
                        'Training Time (s)': [result['training_time']]
                    })], ignore_index=True)
            
            st.dataframe(metrics_df.sort_values(by='R2 Score', ascending=False))
            
            # 可视化比较
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("R2 Score比较" if language == "中文" else "R2 Score Comparison")
                fig = px.bar(
                    metrics_df.sort_values(by='R2 Score', ascending=False),
                    x='Model',
                    y='R2 Score',
                    color='R2 Score',
                    color_continuous_scale='Viridis',
                    labels={'Model': '模型' if language == "中文" else 'Model', 
                           'R2 Score': 'R2分数' if language == "中文" else 'R2 Score'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.subheader("MSE比较" if language == "中文" else "MSE Comparison")
                fig = px.bar(
                    metrics_df.sort_values(by='MSE'),
                    x='Model',
                    y='MSE',
                    color='MSE',
                    color_continuous_scale='Viridis',
                    labels={'Model': '模型' if language == "中文" else 'Model', 
                           'MSE': '均方误差' if language == "中文" else 'MSE'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 选择当前模型进行详细分析
            st.subheader(t("detailed_analysis"))
            selected_model = st.selectbox(
                "选择模型查看详细信息" if language == "中文" else "Select model for detailed analysis", 
                list(st.session_state.models.keys()),
                key="model_detail_select"
            )
            
            if selected_model:
                st.session_state.current_model = selected_model
                model = st.session_state.models[selected_model]
                result = st.session_state.results[selected_model]
                
                st.write(f"### {selected_model} {'性能指标' if language == '中文' else 'Performance Metrics'}")
                col1, col2, col3 = st.columns(3)
                col1.metric("均方误差 (MSE)" if language == "中文" else "Mean Squared Error (MSE)", 
                           f"{result['mse']:.4f}")
                col2.metric("R2 Score", f"{result['r2']:.4f}")
                col3.metric("平均绝对误差 (MAE)" if language == "中文" else "Mean Absolute Error (MAE)", 
                           f"{result['mae']:.4f}")
                
                # 显示模型参数
                st.write(f"### {'模型参数' if language == '中文' else 'Model Parameters'}")
                if hasattr(model, 'best_params_'):
                    st.write(model.best_params_)
                elif hasattr(model, 'get_params'):
                    st.write(model.get_params())
                elif selected_model == t("model_options")[9]:  # LSTM
                    st.text(model.summary())
                
                # 真实值 vs 预测值
                st.write(f"### {'真实值 vs 预测值' if language == '中文' else 'True vs Predicted Values'}")
                
                if STATSMODELS_AVAILABLE:
                    fig = px.scatter(
                        x=result['y_true'],
                        y=result['y_pred'],
                        labels={'x': '真实值' if language == "中文" else 'True values', 
                               'y': '预测值' if language == "中文" else 'Predicted values'},
                        trendline="lowess"
                    )
                else:
                    fig = px.scatter(
                        x=result['y_true'],
                        y=result['y_pred'],
                        labels={'x': '真实值' if language == "中文" else 'True values', 
                               'y': '预测值' if language == "中文" else 'Predicted values'}
                    )
                    st.warning("statsmodels未安装，无法显示趋势线")
                
                fig.add_shape(
                    type="line", line=dict(dash='dash'),
                    x0=min(result['y_true']), y0=min(result['y_true']),
                    x1=max(result['y_true']), y1=max(result['y_true'])
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 残差分析
                st.write(f"### {'残差分析' if language == '中文' else 'Residual Analysis'}")
                residuals = result['y_true'] - result['y_pred']
                
                if STATSMODELS_AVAILABLE:
                    fig = px.scatter(
                        x=result['y_pred'],
                        y=residuals,
                        labels={'x': '预测值' if language == "中文" else 'Predicted values', 
                               'y': '残差' if language == "中文" else 'Residuals'},
                        trendline="lowess"
                    )
                else:
                    fig = px.scatter(
                        x=result['y_pred'],
                        y=residuals,
                        labels={'x': '预测值' if language == "中文" else 'Predicted values', 
                               'y': '残差' if language == "中文" else 'Residuals'}
                    )
                
                fig.add_shape(
                    type="line", line=dict(dash='dash'),
                    x0=min(result['y_pred']), y0=0,
                    x1=max(result['y_pred']), y1=0
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 特征重要性（如果模型支持）
                if selected_model in [t("model_options")[2], t("model_options")[3], t("model_options")[4]]:
                    st.write(f"### {'特征重要性' if language == '中文' else 'Feature Importance'}")
                    if selected_model == t("model_options")[2]:  # 随机森林
                        importances = model.feature_importances_
                    elif selected_model == t("model_options")[3]:  # XGBoost
                        importances = model.feature_importances_
                    else:  # LightGBM
                        importances = model.feature_importances_
                    
                    importance_df = pd.DataFrame({
                        'Feature': st.session_state.feature_cols,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False)
                    
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        color='Importance',
                        color_continuous_scale='Viridis',
                        labels={'Feature': '特征' if language == "中文" else 'Feature', 
                               'Importance': '重要性' if language == "中文" else 'Importance'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # LSTM训练历史
                if selected_model == t("model_options")[9]:
                    history = st.session_state.results["LSTM网络_history" if language == "中文" else "LSTM Network_history"]
                    st.write(f"### {'训练历史' if language == '中文' else 'Training History'}")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=history['loss'],
                        name='训练损失' if language == "中文" else 'Training loss'
                    ))
                    fig.add_trace(go.Scatter(
                        y=history['val_loss'],
                        name='验证损失' if language == "中文" else 'Validation loss'
                    ))
                    fig.update_layout(
                        title='训练和验证损失' if language == "中文" else 'Training and Validation Loss',
                        xaxis_title='Epoch',
                        yaxis_title='Loss' if language == "English" else '损失'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header(t("visualization"))
        
        if not st.session_state.models or not st.session_state.current_model:
            st.warning("请先训练并选择一个模型!" if language == "中文" else "Please train and select a model first!")
        else:
            selected_model = st.session_state.current_model
            result = st.session_state.results[selected_model]
            
            # 3D可视化
            if len(st.session_state.feature_cols) >= 2:
                st.subheader(t("3d_plot"))
                
                # 选择两个特征进行3D可视化
                col1, col2 = st.columns(2)
                with col1:
                    x_feature = st.selectbox(
                        "X轴特征" if language == "中文" else "X Feature", 
                        st.session_state.feature_cols, index=0, key="3d_x_feature"
                    )
                with col2:
                    y_feature = st.selectbox(
                        "Y轴特征" if language == "中文" else "Y Feature", 
                        st.session_state.feature_cols, index=1, key="3d_y_feature"
                    )
                
                # 创建网格数据
                x_min, x_max = st.session_state.X_test[x_feature].min(), st.session_state.X_test[x_feature].max()
                y_min, y_max = st.session_state.X_test[y_feature].min(), st.session_state.X_test[y_feature].max()
                
                x_range = np.linspace(x_min, x_max, 20)
                y_range = np.linspace(y_min, y_max, 20)
                xx, yy = np.meshgrid(x_range, y_range)
                
                # 创建预测网格
                if len(st.session_state.feature_cols) == 2:
                    grid_data = np.c_[xx.ravel(), yy.ravel()]
                else:
                    # 对其他特征使用中位数
                    other_features = [f for f in st.session_state.feature_cols if f not in [x_feature, y_feature]]
                    median_values = st.session_state.X_test[other_features].median().values
                    grid_data = []
                    for x, y in zip(xx.ravel(), yy.ravel()):
                        row = [x, y] + list(median_values)
                        grid_data.append(row)
                    grid_data = np.array(grid_data)
                
                # 预测
                model = st.session_state.models[selected_model]
                if selected_model == t("model_options")[9]:  # LSTM
                    # 对于LSTM，我们需要创建时间序列数据
                    # 这里简化处理，可能不太准确
                    time_steps = 3
                    lstm_grid_data = []
                    for i in range(time_steps, len(grid_data)):
                        lstm_grid_data.append(grid_data[i-time_steps:i])
                    lstm_grid_data = np.array(lstm_grid_data)
                    zz = model.predict(lstm_grid_data).reshape(xx.shape)
                else:
                    zz = model.predict(grid_data).reshape(xx.shape)
                
                # 创建3D图
                fig = go.Figure()
                
                # 添加预测表面
                fig.add_trace(go.Surface(
                    x=xx,
                    y=yy,
                    z=zz,
                    colorscale='Viridis',
                    opacity=0.7,
                    name='预测表面' if language == "中文" else 'Prediction surface'
                ))
                
                # 添加真实数据点
                fig.add_trace(go.Scatter3d(
                    x=st.session_state.X_test[x_feature],
                    y=st.session_state.X_test[y_feature],
                    z=result['y_true'],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=result['y_true'],
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    name='真实值' if language == "中文" else 'True values'
                ))
                
                # 添加预测数据点
                fig.add_trace(go.Scatter3d(
                    x=st.session_state.X_test[x_feature],
                    y=st.session_state.X_test[y_feature],
                    z=result['y_pred'],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=result['y_pred'],
                        colorscale='Viridis',
                        opacity=0.8,
                        symbol='x'
                    ),
                    name='预测值' if language == "中文" else 'Predicted values'
                ))
                
                fig.update_layout(
                    scene=dict(
                        xaxis_title=x_feature,
                        yaxis_title=y_feature,
                        zaxis_title=st.session_state.target_col
                    ),
                    title=f"{selected_model} 3D预测可视化" if language == "中文" else f"{selected_model} 3D Prediction Visualization",
                    width=1000,
                    height=800
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # 时间序列分析（如果数据有时间特征）
            st.subheader(t("time_series"))
            
            if 'date' in st.session_state.data.columns or 'time' in st.session_state.data.columns:
                time_col = 'date' if 'date' in st.session_state.data.columns else 'time'
                st.write(f"使用 '{time_col}' 列作为时间轴" if language == "中文" else f"Using '{time_col}' column as time axis")
                
                time_series_data = st.session_state.data.copy()
                time_series_data[time_col] = pd.to_datetime(time_series_data[time_col])
                time_series_data = time_series_data.sort_values(time_col)
                
                fig = px.line(
                    time_series_data,
                    x=time_col,
                    y=st.session_state.target_col,
                    title='目标变量时间序列' if language == "中文" else 'Target Variable Time Series'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("未检测到时间特征列 (如 'date' 或 'time')" if language == "中文" else "No time feature column detected (like 'date' or 'time')")
    
    with tab5:
        st.header(t("model_saving"))
        
        if not st.session_state.models:
            st.warning("没有可保存的模型!" if language == "中文" else "No models to save!")
        else:
            st.subheader(t("save_model"))
            selected_model_to_save = st.selectbox(
                "选择要保存的模型" if language == "中文" else "Select model to save", 
                list(st.session_state.models.keys()),
                key="model_save_select"
            )
            
            if st.button(t("save_model"), key="save_button"):
                model = st.session_state.models[selected_model_to_save]
                filename = f"{selected_model_to_save.replace(' ', '_')}_model.joblib"
                joblib.dump(model, filename)
                st.success(f"模型已保存为 {filename}" if language == "中文" else f"Model saved as {filename}")
                
                # 提供下载链接
                with open(filename, "rb") as f:
                    st.download_button(
                        "下载模型文件" if language == "中文" else "Download Model File",
                        f,
                        file_name=filename
                    )
            
            st.subheader(t("export_results"))
            if st.button("导出所有结果到CSV" if language == "中文" else "Export all results to CSV", key="export_csv_button"):
                # 准备结果数据
                all_results = []
                for model_name, result in st.session_state.results.items():
                    if "_history" not in model_name:  # 排除LSTM的训练历史
                        row = {
                            'Model': model_name,
                            'MSE': result['mse'],
                            'R2': result['r2'],
                            'MAE': result['mae'],
                            'TrainingTime': result['training_time']
                        }
                        all_results.append(row)
                
                results_df = pd.DataFrame(all_results)
                csv = results_df.to_csv(index=False)
                
                # 提供下载
                st.download_button(
                    "下载结果CSV" if language == "中文" else "Download Results CSV",
                    csv,
                    file_name="model_results.csv",
                    mime="text/csv"
                )
            
            st.subheader(t("export_charts"))
            
            # 导出设置区域
            col1, col2 = st.columns(2)
            
            with col1:
                # 移除导出格式选择，始终使用HTML
                st.info("图表将以HTML格式导出" if language == "中文" else "Charts will be exported in HTML format")
                export_format = "html"  # 强制使用HTML格式
            
            with col2:
                # 导出位置选择
                export_folder = st.text_input(
                    "导出文件位置" if language == "中文" else "Export Directory",
                    "charts",
                    key="export_folder"
                )
                # 添加浏览按钮选项的提示
                st.caption("输入文件夹路径，可以是相对路径或绝对路径" if language == "中文" else 
                          "Enter folder path, can be relative or absolute path")
            
            # 添加选择导出文件位置的高级选项
            with st.expander("高级导出选项" if language == "中文" else "Advanced Export Options"):
                use_timestamped_folder = st.checkbox(
                    "为每次导出创建时间戳文件夹" if language == "中文" else "Create timestamped folder for each export",
                    True,
                    key="use_timestamped_folder"
                )
                
                # 添加选项让用户选择文件名前缀
                file_prefix = st.text_input(
                    "文件名前缀" if language == "中文" else "Filename Prefix",
                    "model_analysis",
                    key="file_prefix"
                )
            
            # 添加数据导出选项
            col1, col2 = st.columns(2)
            
            with col1:
                # Excel导出选项
                include_excel = st.checkbox(
                    "导出详细结果(Excel)" if language == "中文" else "Export detailed results (Excel)", 
                    True,
                    key="include_excel"
                )
            
            with col2:
                # 为每张图导出CSV数据
                export_chart_data = st.checkbox(
                    "为每张图导出数据(CSV)" if language == "中文" else "Export data for each chart (CSV)",
                    True,
                    key="export_chart_data"
                )
            
            # 更新导出按钮处理逻辑
            if st.button(t("export_all_charts"), key="export_charts_button"):
                # 创建时间戳
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 创建导出文件夹，根据用户选择决定是否使用时间戳
                if use_timestamped_folder:
                    export_path = os.path.join(export_folder, f"{file_prefix}_{timestamp}")
                else:
                    export_path = export_folder
                
                # 尝试创建导出目录
                try:
                    os.makedirs(export_path, exist_ok=True)
                    st.info(f"{'导出到目录:' if language == '中文' else 'Exporting to directory:'} {export_path}")
                except Exception as e:
                    st.error(f"{'创建导出目录失败:' if language == '中文' else 'Failed to create export directory:'} {str(e)}")
                    st.stop()  # 如果无法创建目录，停止执行
                
                # 强制使用HTML格式
                export_format = "html"
                
                # 跟踪已成功导出的文件
                exported_files = []
                exported_data_files = []
                
                # 准备Excel数据
                if include_excel:
                    # 创建Excel writer
                    excel_path = os.path.join(export_path, f"{file_prefix}_results_{timestamp}.xlsx")
                    excel_writer = pd.ExcelWriter(excel_path, engine='openpyxl')
                    
                    # 创建汇总表
                    summary_df = pd.DataFrame(columns=['Model', 'MSE', 'R2 Score', 'MAE', 'Training Time (s)'])
                    
                    # 跟踪所有详细预测数据
                    all_predictions = {}
                
                # 创建导出进度条
                export_progress = st.progress(0)
                export_status = st.empty()
                
                # 计算总图表数量 - 考虑特征重要性图表可能存在
                model_count = len([name for name in st.session_state.results if "_history" not in name])
                # 每个模型至少2张图 (真实vs预测、残差)，某些模型可能有特征重要性图
                total_charts = model_count * 3  # 使用较大的数字以确保进度不超过1.0
                chart_count = 0
                
                # 导出每个模型的详细图表
                for model_name, result in st.session_state.results.items():
                    if "_history" not in model_name:  # 排除模型训练历史
                        export_status.text(f"正在导出 {model_name} 的图表..." if language == "中文" else f"Exporting charts for {model_name}...")
                        
                        # 将预测数据添加到Excel中
                        if include_excel:
                            # 添加到汇总表
                            summary_df = pd.concat([summary_df, pd.DataFrame({
                                'Model': [model_name],
                                'MSE': [result['mse']],
                                'R2 Score': [result['r2']],
                                'MAE': [result['mae']],
                                'Training Time (s)': [result['training_time']]
                            })], ignore_index=True)
                            
                            # 为每个模型创建详细预测数据表
                            pred_df = pd.DataFrame({
                                'True Value': result['y_true'],
                                'Predicted Value': result['y_pred'],
                                'Error': result['y_true'] - result['y_pred']
                            })
                            all_predictions[model_name] = pred_df
                        
                        # 真实值vs预测值图表
                        fig = px.scatter(
                            x=result['y_true'],
                            y=result['y_pred'],
                            labels={'x': '真实值' if language == "中文" else 'True values', 
                                   'y': '预测值' if language == "中文" else 'Predicted values'},
                            title=f"{model_name} - {'真实值 vs 预测值' if language == '中文' else 'True vs Predicted Values'}"
                        )
                        fig.add_shape(
                            type="line", line=dict(dash='dash'),
                            x0=min(result['y_true']), y0=min(result['y_true']),
                            x1=max(result['y_true']), y1=max(result['y_true'])
                        )
                        
                        # 准备图表数据
                        chart_data = {
                            'true_values': result['y_true'],
                            'predicted_values': result['y_pred']
                        } if export_chart_data else None
                        
                        export_result = save_chart(
                            fig, 
                            f"{model_name}_pred_vs_true_{timestamp}", 
                            data=chart_data,
                            folder=export_path,
                            format=export_format
                        )
                        
                        if export_result["success"]:
                            exported_files.append(export_result["filepath"])
                            if export_result["data_filepath"]:
                                exported_data_files.append(export_result["data_filepath"])
                        
                        chart_count += 1
                        export_progress.progress(min(chart_count / total_charts, 0.99))
                        
                        # 残差分析图表
                        residuals = result['y_true'] - result['y_pred']
                        fig = px.scatter(
                            x=result['y_pred'],
                            y=residuals,
                            labels={'x': '预测值' if language == "中文" else 'Predicted values', 
                                   'y': '残差' if language == "中文" else 'Residuals'},
                            title=f"{model_name} - {'残差分析' if language == '中文' else 'Residual Analysis'}"
                        )
                        fig.add_shape(
                            type="line", line=dict(dash='dash'),
                            x0=min(result['y_pred']), y0=0,
                            x1=max(result['y_pred']), y1=0
                        )
                        
                        # 准备残差图表数据
                        residual_data = {
                            'predicted_values': result['y_pred'],
                            'residuals': residuals
                        } if export_chart_data else None
                        
                        export_result = save_chart(
                            fig, 
                            f"{model_name}_residuals_{timestamp}", 
                            data=residual_data,
                            folder=export_path,
                            format=export_format
                        )
                        
                        if export_result["success"]:
                            exported_files.append(export_result["filepath"])
                            if export_result["data_filepath"]:
                                exported_data_files.append(export_result["data_filepath"])
                        
                        chart_count += 1
                        export_progress.progress(min(chart_count / total_charts, 0.99))
                        
                        # 为支持的模型导出特征重要性图表
                        if model_name in [t("model_options")[2], t("model_options")[3], t("model_options")[4], t("model_options")[10]]:
                            if hasattr(st.session_state.models[model_name], 'feature_importances_'):
                                importances = st.session_state.models[model_name].feature_importances_
                                importance_df = pd.DataFrame({
                                    'Feature': st.session_state.feature_cols,
                                    'Importance': importances
                                }).sort_values(by='Importance', ascending=False)
                                
                                fig = px.bar(
                                    importance_df,
                                    x='Importance',
                                    y='Feature',
                                    orientation='h',
                                    color='Importance',
                                    color_continuous_scale='Viridis',
                                    title=f"{model_name} - {'特征重要性' if language == '中文' else 'Feature Importance'}",
                                    labels={'Feature': '特征' if language == "中文" else 'Feature', 
                                           'Importance': '重要性' if language == "中文" else 'Importance'}
                                )
                                
                                export_result = save_chart(
                                    fig, 
                                    f"{model_name}_feature_importance_{timestamp}", 
                                    data=importance_df if export_chart_data else None,
                                    folder=export_path,
                                    format=export_format
                                )
                                
                                if export_result["success"]:
                                    exported_files.append(export_result["filepath"])
                                    if export_result["data_filepath"]:
                                        exported_data_files.append(export_result["data_filepath"])
                                
                                chart_count += 1
                                export_progress.progress(min(chart_count / total_charts, 0.99))
                                
                                # 存储特征重要性数据到Excel
                                if include_excel:
                                    all_predictions[f"{model_name}_importance"] = importance_df
                
                # 导出模型性能比较图表
                if len(st.session_state.results) > 1:
                    export_status.text("正在导出模型性能比较图表..." if language == "中文" else "Exporting model performance comparison charts...")
                    
                    metrics_df = pd.DataFrame(columns=['Model', 'MSE', 'R2 Score', 'MAE'])
                    for model_name, result in st.session_state.results.items():
                        if "_history" not in model_name:
                            metrics_df = pd.concat([metrics_df, pd.DataFrame({
                                'Model': [model_name],
                                'MSE': [result['mse']],
                                'R2 Score': [result['r2']],
                                'MAE': [result['mae']]
                            })], ignore_index=True)
                    
                    # R2比较图表
                    fig = px.bar(
                        metrics_df.sort_values(by='R2 Score', ascending=False),
                        x='Model',
                        y='R2 Score',
                        color='R2 Score',
                        color_continuous_scale='Viridis',
                        title='R2 Score 比较' if language == "中文" else 'R2 Score Comparison'
                    )
                    
                    export_result = save_chart(
                        fig, 
                        f"r2_comparison_{timestamp}", 
                        data=metrics_df[['Model', 'R2 Score']].sort_values(by='R2 Score', ascending=False) if export_chart_data else None,
                        folder=export_path,
                        format=export_format
                    )
                    
                    if export_result["success"]:
                        exported_files.append(export_result["filepath"])
                        if export_result["data_filepath"]:
                            exported_data_files.append(export_result["data_filepath"])
                    
                    # MSE比较图表
                    fig = px.bar(
                        metrics_df.sort_values(by='MSE'),
                        x='Model',
                        y='MSE',
                        color='MSE',
                        color_continuous_scale='Viridis',
                        title='MSE 比较' if language == "中文" else 'MSE Comparison'
                    )
                    
                    export_result = save_chart(
                        fig, 
                        f"mse_comparison_{timestamp}", 
                        data=metrics_df[['Model', 'MSE']].sort_values(by='MSE') if export_chart_data else None,
                        folder=export_path,
                        format=export_format
                    )
                    
                    if export_result["success"]:
                        exported_files.append(export_result["filepath"])
                        if export_result["data_filepath"]:
                            exported_data_files.append(export_result["data_filepath"])
                    
                    # 导出MAE比较图表
                    fig = px.bar(
                        metrics_df.sort_values(by='MAE'),
                        x='Model',
                        y='MAE',
                        color='MAE',
                        color_continuous_scale='Viridis',
                        title='MAE 比较' if language == "中文" else 'MAE Comparison'
                    )
                    
                    export_result = save_chart(
                        fig, 
                        f"mae_comparison_{timestamp}", 
                        data=metrics_df[['Model', 'MAE']].sort_values(by='MAE') if export_chart_data else None,
                        folder=export_path,
                        format=export_format
                    )
                    
                    if export_result["success"]:
                        exported_files.append(export_result["filepath"])
                        if export_result["data_filepath"]:
                            exported_data_files.append(export_result["data_filepath"])
                
                # 导出深度学习模型的训练历史图表
                for model_name in [t("model_options")[9], t("model_options")[12], t("model_options")[13]]:
                    history_key = model_name + "_history"
                    if history_key in st.session_state.results:
                        export_status.text(f"正在导出 {model_name} 的训练历史..." if language == "中文" else f"Exporting training history for {model_name}...")
                        
                        history = st.session_state.results[history_key]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=history['loss'],
                            name='训练损失' if language == "中文" else 'Training loss'
                        ))
                        fig.add_trace(go.Scatter(
                            y=history['val_loss'],
                            name='验证损失' if language == "中文" else 'Validation loss'
                        ))
                        fig.update_layout(
                            title=f"{model_name} - {'训练历史' if language == '中文' else 'Training History'}",
                            xaxis_title='Epoch',
                            yaxis_title='Loss' if language == "English" else '损失'
                        )
                        
                        # 准备训练历史数据
                        history_df = pd.DataFrame({
                            'epoch': range(1, len(history['loss'])+1),
                            'training_loss': history['loss'],
                            'validation_loss': history['val_loss'] if 'val_loss' in history else None
                        })
                        
                        export_result = save_chart(
                            fig, 
                            f"{model_name}_training_history_{timestamp}", 
                            data=history_df if export_chart_data else None,
                            folder=export_path,
                            format=export_format
                        )
                        
                        if export_result["success"]:
                            exported_files.append(export_result["filepath"])
                            if export_result["data_filepath"]:
                                exported_data_files.append(export_result["data_filepath"])
                        
                        # 添加训练历史数据到Excel
                        if include_excel:
                            all_predictions[f"{model_name}_training_history"] = history_df
                
                # 导出Excel文件
                if include_excel and 'summary_df' in locals() and summary_df.shape[0] > 0:
                    export_status.text("正在导出Excel数据..." if language == "中文" else "Exporting Excel data...")
                    
                    try:
                        # 写入模型性能汇总表
                        summary_df.to_excel(excel_writer, sheet_name='Models_Summary', index=False)
                        
                        # 写入每个模型的详细预测数据
                        for model_name, pred_df in all_predictions.items():
                            # 如果表名太长，截断它
                            sheet_name = model_name[:31] if len(model_name) > 31 else model_name
                            pred_df.to_excel(excel_writer, sheet_name=sheet_name, index=False)
                        
                        # 保存Excel文件
                        excel_writer.close()
                        exported_files.append(excel_path)
                    except Exception as e:
                        st.error(f"导出Excel文件失败: {str(e)}" if language == "中文" else f"Failed to export Excel file: {str(e)}")
                
                # 完成进度条
                export_progress.progress(1.0)
                export_status.empty()
                
                if exported_files:
                    # 创建导出清单文件，便于查看所有导出内容
                    try:
                        export_manifest = {
                            "timestamp": timestamp,
                            "charts": exported_files,
                            "data_files": exported_data_files,
                            "excel_file": excel_path if include_excel else None
                        }
                        
                        with open(f"{export_path}/export_manifest.json", "w") as f:
                            json.dump(export_manifest, f, indent=2)
                    except:
                        pass
                    
                    st.success(f"成功导出 {len(exported_files)} 个文件到 {export_path} 文件夹!" if language == "中文" 
                              else f"Successfully exported {len(exported_files)} files to {export_path} folder!")
                    
                    if export_chart_data:
                        st.success(f"已为 {len(exported_data_files)} 张图表导出CSV数据文件，可在Origin中使用!" if language == "中文"
                                  else f"CSV data files exported for {len(exported_data_files)} charts, ready for use in Origin!")
                    
                    # 显示导出的文件列表
                    with st.expander("查看导出的文件" if language == "中文" else "View exported files"):
                        st.write("**图表文件:**" if language == "中文" else "**Chart files:**")
                        for file in exported_files:
                            if not file.endswith('.xlsx') and not file.endswith('.csv'):
                                st.write(f"- {file}")
                        
                        if export_chart_data and exported_data_files:
                            st.write("**数据文件 (用于Origin):**" if language == "中文" else "**Data files (for Origin):**")
                            for file in exported_data_files:
                                st.write(f"- {file}")
                        
                        if include_excel and 'excel_path' in locals():
                            st.write("**Excel结果文件:**" if language == "中文" else "**Excel results file:**")
                            st.write(f"- {excel_path}")
                    
                    # 提供下载整个文件夹的链接（需要先压缩）
                    try:
                        import shutil
                        zip_filename = f"{file_prefix}_export_{timestamp}.zip"
                        zip_path = os.path.join(os.path.dirname(export_path), zip_filename)
                        
                        # 创建ZIP文件
                        shutil.make_archive(
                            os.path.splitext(zip_path)[0],  # 移除.zip后缀，make_archive会自动添加
                            'zip', 
                            export_path
                        )
                        
                        with open(zip_path, "rb") as f:
                            st.download_button(
                                "下载所有导出文件 (ZIP)" if language == "中文" else "Download all exported files (ZIP)",
                                f,
                                file_name=zip_filename,
                                mime="application/zip"
                            )
                    except Exception as e:
                        st.error(f"{'创建ZIP文件失败:' if language == '中文' else 'Failed to create ZIP file:'} {str(e)}")
                else:
                    st.error("导出图表失败，请检查控制台错误信息" if language == "中文" else "Failed to export charts. Please check console for errors")
                    st.info("尝试安装或更新必要的库: pip install kaleido plotly --force-reinstall" if language == "中文" 
                           else "Try installing or updating required libraries: pip install kaleido plotly --force-reinstall")
                
else:
    st.info("请上传数据集并完成预处理步骤" if language == "中文" else "Please upload dataset and complete preprocessing steps")