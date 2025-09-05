import pandas as pd
import os

# 切换到附件目录
os.chdir('CUMCM2025Problems/A题/附件/')

# 读取三个Excel文件
print("=== 读取Excel文件 ===")
try:
    df1 = pd.read_excel('result1.xlsx')
    print("result1.xlsx:")
    print("形状:", df1.shape)
    print("列名:", df1.columns.tolist())
    print("前5行:")
    print(df1.head())
    print("\n" + "="*50 + "\n")
except Exception as e:
    print("读取result1.xlsx出错:", e)

try:
    df2 = pd.read_excel('result2.xlsx')
    print("result2.xlsx:")
    print("形状:", df2.shape)
    print("列名:", df2.columns.tolist())
    print("前5行:")
    print(df2.head())
    print("\n" + "="*50 + "\n")
except Exception as e:
    print("读取result2.xlsx出错:", e)

try:
    df3 = pd.read_excel('result3.xlsx')
    print("result3.xlsx:")
    print("形状:", df3.shape)
    print("列名:", df3.columns.tolist())
    print("前5行:")
    print(df3.head())
    print("\n" + "="*50 + "\n")
except Exception as e:
    print("读取result3.xlsx出错:", e)
