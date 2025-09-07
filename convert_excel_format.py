#!/usr/bin/env python3
"""
Excel格式转换工具
将result1_v6.xlsx格式转换为result1_v5.xlsx格式

使用方法:
python convert_excel_format.py [输入文件] [输出文件]

如果不指定参数，默认转换result1_v6.xlsx
"""
import pandas as pd
import numpy as np
import sys
import os

def convert_v6_to_v5_format(input_file="result1_v6.xlsx", output_file=None):
    """
    将v6格式的Excel文件转换为v5格式
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径（如果为None，则覆盖输入文件）
    """
    
    if output_file is None:
        output_file = input_file
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在！")
        return False
    
    try:
        # 读取输入文件
        print(f"正在读取 {input_file}...")
        df_v6 = pd.read_excel(input_file, sheet_name="Sheet1")
        
        print("原始v6格式:")
        print(f"列名: {list(df_v6.columns)}")
        print(f"形状: {df_v6.shape}")
        print("前几行数据:")
        print(df_v6.head())
        
        # 创建新的DataFrame，按照v5格式的列顺序
        df_v5_format = pd.DataFrame()
        
        # 列映射和转换
        print("\n正在进行格式转换...")
        
        # 1. v6 '无人机航向(rad)' -> v5 '无人机运动方向' (添加rad单位)
        if '无人机航向(rad)' in df_v6.columns:
            df_v5_format['无人机运动方向'] = df_v6['无人机航向(rad)'].apply(lambda x: f"{x:.6f} rad")
        else:
            print("警告：未找到'无人机航向(rad)'列")
        
        # 2. v6 '无人机速度(m/s)' -> v5 '无人机运动速度 (m/s)'
        if '无人机速度(m/s)' in df_v6.columns:
            df_v5_format['无人机运动速度 (m/s)'] = df_v6['无人机速度(m/s)']
        else:
            print("警告：未找到'无人机速度(m/s)'列")
        
        # 3. v6 '弹序号' -> v5 '烟幕干扰弹编号' (转换中文序号为数字)
        if '弹序号' in df_v6.columns:
            bomb_number_map = {
                "第一枚": 1, "第二枚": 2, "第三枚": 3, "第四枚": 4, "第五枚": 5,
                "第六枚": 6, "第七枚": 7, "第八枚": 8, "第九枚": 9, "第十枚": 10
            }
            df_v5_format['烟幕干扰弹编号'] = df_v6['弹序号'].map(bomb_number_map)
            # 如果映射失败，使用行索引+1作为编号
            mask = df_v5_format['烟幕干扰弹编号'].isna()
            df_v5_format.loc[mask, '烟幕干扰弹编号'] = df_v6.loc[mask].index + 1
        else:
            print("警告：未找到'弹序号'列")
        
        # 4-6. 投放点坐标转换
        coord_mapping = [
            ('投放点X(m)', '烟幕干扰弹投放点的x坐标 (m)'),
            ('投放点Y(m)', '烟幕干扰弹投放点的y坐标 (m)'),
            ('投放点Z(m)', '烟幕干扰弹投放点的z坐标 (m)')
        ]
        
        for v6_col, v5_col in coord_mapping:
            if v6_col in df_v6.columns:
                df_v5_format[v5_col] = df_v6[v6_col]
            else:
                print(f"警告：未找到'{v6_col}'列")
        
        # 7-9. 起爆点坐标转换
        explosion_mapping = [
            ('起爆点X(m)', '烟幕干扰弹起爆点的x坐标 (m)'),
            ('起爆点Y(m)', '烟幕干扰弹起爆点的y坐标 (m)'),
            ('起爆点Z(m)', '烟幕干扰弹起爆点的z坐标 (m)')
        ]
        
        for v6_col, v5_col in explosion_mapping:
            if v6_col in df_v6.columns:
                df_v5_format[v5_col] = df_v6[v6_col]
            else:
                print(f"警告：未找到'{v6_col}'列")
        
        # 10. 有效干扰时长 (v6格式中没有直接对应，设为默认值0.00)
        df_v5_format['有效干扰时长 (s)'] = 0.00
        
        print("\n转换后的v5格式:")
        print(f"列名: {list(df_v5_format.columns)}")
        print(f"形状: {df_v5_format.shape}")
        print("前几行数据:")
        print(df_v5_format.head())
        
        # 保存转换后的文件
        print(f"\n正在保存到 {output_file}...")
        df_v5_format.to_excel(output_file, sheet_name="Sheet1", index=False)
        print(f"成功将 {input_file} 转换为v5格式并保存到 {output_file}！")
        
        return True
        
    except Exception as e:
        print(f"转换过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def examine_excel_file(filename):
    """检查Excel文件的格式"""
    print(f"\n=== 检查 {filename} ===")
    
    try:
        if not os.path.exists(filename):
            print(f"文件 {filename} 不存在")
            return
            
        xl_file = pd.ExcelFile(filename)
        print(f"工作表: {xl_file.sheet_names}")
        
        for sheet_name in xl_file.sheet_names:
            print(f"\n--- 工作表: {sheet_name} ---")
            df = pd.read_excel(filename, sheet_name=sheet_name)
            print(f"形状: {df.shape}")
            print(f"列名: {list(df.columns)}")
            print("前几行:")
            print(df.head())
            print("数据类型:")
            print(df.dtypes)
            
    except Exception as e:
        print(f"读取 {filename} 时出错: {e}")

def main():
    """主函数"""
    if len(sys.argv) == 1:
        # 默认转换result1_v6.xlsx
        print("使用默认参数转换 result1_v6.xlsx")
        convert_v6_to_v5_format()
    elif len(sys.argv) == 2:
        if sys.argv[1] in ['-h', '--help', 'help']:
            print(__doc__)
            return
        elif sys.argv[1] == 'examine':
            # 检查两个文件的格式
            examine_excel_file("result1_v5.xlsx")
            examine_excel_file("result1_v6.xlsx")
            return
        else:
            # 转换指定文件
            input_file = sys.argv[1]
            convert_v6_to_v5_format(input_file)
    elif len(sys.argv) == 3:
        # 转换指定输入文件到指定输出文件
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        convert_v6_to_v5_format(input_file, output_file)
    else:
        print("参数错误！")
        print(__doc__)

if __name__ == "__main__":
    main()
