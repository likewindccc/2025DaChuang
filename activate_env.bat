@echo off
echo 激活农村女性就业研究项目环境...
call venv\Scripts\activate.bat
echo 项目环境已激活！
echo 当前Python路径：
python -c "import sys; print(sys.executable)"
echo 已安装的主要包：
echo - numpy, pandas, scipy (数据分析)
echo - matplotlib, seaborn (可视化)
echo - scikit-learn, statsmodels (机器学习/统计)  
echo - quantecon (经济学建模)
echo - networkx (网络分析)
cmd /k
