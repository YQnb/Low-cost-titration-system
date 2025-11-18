import itertools
import os
import pandas as pd
from data_operate import titration_curve
import numpy as np

# 基础组分定义
BASE_COMPONENTS = {
    'acetate': {
        'pKa': 4.76,
        'initial_form': 'A-',
        'charge_A': -1,
        'charge_HA': 0,
        'counter_ion_charge': 1,
        'ifaddition': 1
    },
    'phosphate': {
        'pKa': 7.21,
        'initial_form': 'HA',
        'charge_A': -2,
        'charge_HA': -1,
        'counter_ion_charge': 1,
        'ifaddition': 1
    },
    'ammonium': {
        'pKa': 9.25,
        'initial_form': 'HA',
        'charge_A': 0,
        'charge_HA': 1,
        'counter_ion_charge': -1,
        'ifaddition': 1
    },
    'citrate': {
        'pKa': 6.40,
        'initial_form': 'A-',
        'charge_A': -3,
        'charge_HA': -2,
        'counter_ion_charge': 1,
        'ifaddition': 1
    }
}

# 混合缓冲液配方 (所有可能的双组分组合)
MIXED_BUFFER_RECIPES = {
    # 酸性缓冲液组合
    'acetate_phosphate': ['acetate', 'phosphate'],
    'acetate_ammonium': ['acetate', 'ammonium'],
    'acetate_citrate': ['acetate', 'citrate'],

    # 磷酸盐相关组合
    'phosphate_ammonium': ['phosphate', 'ammonium'],
    'phosphate_citrate': ['phosphate', 'citrate'],

    # 其他组合
    'ammonium_citrate': ['ammonium', 'citrate']
}

# CONC_OPTIONS = [0.01, 0.05, 0.1, 0.2]
CONC_OPTIONS = np.arange(0.01, 0.5, 0.07)
INITIAL_VOLUME = 30.0  # mL
TITRANT_CONC = 0.05  # M

def generate_single_component_curves():
    """生成单组分缓冲液的滴定曲线"""
    os.makedirs('train_csv/single', exist_ok=True)
    all_dfs = []

    for comp_name, params in BASE_COMPONENTS.items():
        # 遍历所有浓度
        for conc in CONC_OPTIONS:
            system = {
                'species': [{
                    'name': comp_name,
                    'pKa': params['pKa'],
                    'total_conc': conc,
                    'initial_form': params['initial_form'],
                    'charge_A': params['charge_A'],
                    'charge_HA': params['charge_HA'],
                    'counter_ion_charge': params['counter_ion_charge'],
                    'ifaddition': params['ifaddition']
                }]
            }

            # 确定滴定方向
            titrant_charge = 1 if params['initial_form'] == 'HA' else -1

            df = titration_curve(
                system,
                titrant_conc=TITRANT_CONC,
                titrant_charge=titrant_charge,
                initial_volume=INITIAL_VOLUME,
                step_size=0.1
            )

            # 添加元数据
            df['Component_Type'] = 'single'
            df['Component_Name'] = comp_name
            df['Concentration_M'] = conc
            all_dfs.append(df)

    # 保存单组分数据
    single_df = pd.concat(all_dfs, ignore_index=True)
    single_df.to_csv('train_csv/single/phosphate_ammonium.csv', index=False)
    print(f"已生成 {len(single_df)} 行单组分数据")
    return single_df

def generate_mixed_buffer_curves():
    """生成混合缓冲液的滴定曲线"""
    os.makedirs('train_csv/mixed', exist_ok=True)
    all_dfs = []

    for mix_name, components in MIXED_BUFFER_RECIPES.items():
        # 生成所有浓度组合
        for conc_comb in itertools.product(CONC_OPTIONS, repeat=len(components)):
            system = {'species': []}

            # 构建混合系统
            for comp_name, conc in zip(components, conc_comb):
                comp = BASE_COMPONENTS[comp_name].copy()
                comp.update({
                    'name': comp_name,
                    'total_conc': conc,
                    'ifaddition': 1  # 混合缓冲液默认带反离子
                })
                system['species'].append(comp)

            # 确定滴定方向
            has_acid = any(c['initial_form'] == 'HA' for c in system['species'])
            titrant_charge = 1 if has_acid else -1

            df = titration_curve(
                system,
                titrant_conc=TITRANT_CONC,
                titrant_charge=titrant_charge,
                initial_volume=INITIAL_VOLUME,
                step_size=0.1
            )

            # 添加元数据
            df['Component_Type'] = 'mixed'
            df['Buffer_Mix'] = mix_name
            for i, (comp, conc) in enumerate(zip(components, conc_comb), 1):
                df[f'Component_{i}_Name'] = comp
                df[f'Component_{i}_Conc_M'] = conc

            all_dfs.append(df)

    # 保存混合缓冲液数据
    mixed_df = pd.concat(all_dfs, ignore_index=True)
    mixed_df.to_csv('train_csv/mixed/phosphate_ammonium.csv', index=False)
    print(f"已生成 {len(mixed_df)} 行混合缓冲液数据")
    return mixed_df

def merge_and_save_datasets(single_df, mixed_df):
    """合并数据集并保存"""
    # 统一列名
    for df in [single_df, mixed_df]:
        if 'Component_Name' not in df:
            df['Component_Name'] = None

    # 合并数据
    full_df = pd.concat([single_df, mixed_df], ignore_index=True)

    # 重新组织列顺序
    base_columns = ['Volume_mL', 'pH', 'Ionic_Strength', 'Buffer_Capacity',
                   'Component_Type', 'Component_Name', 'Concentration_M', 'Buffer_Mix']
    component_columns = [c for c in full_df.columns if c.startswith('Component_') and c not in base_columns]
    full_df = full_df[base_columns + component_columns]

    # 保存完整数据集
    full_df.to_csv('train_csv/phosphate_ammonium.csv', index=False)
    print(f"完整数据集已保存，总计 {len(full_df)} 行数据")
    return full_df

if __name__ == '__main__':
    # 生成单组分和混合缓冲液数据
    single_df = generate_single_component_curves()
    mixed_df = generate_mixed_buffer_curves()

    # 合并并保存完整数据集
    full_df = merge_and_save_datasets(single_df, mixed_df)