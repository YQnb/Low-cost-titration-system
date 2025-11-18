import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.optimize import brentq

# ------------------------- 常数 -------------------------
Kw = 1e-14  # 水的离子积

# ------------------------- 工具函数 -------------------------
def calculate_pH(H_conc):
    """计算pH值，处理极值"""
    return -np.log10(max(min(H_conc, 1.0), 1e-14))

def charge_balance(H_conc, species, additional_cations=[], additional_anions=[]):
    """
    通用电荷平衡方程
    参数:
        H_conc: H+浓度
        species: 带电物种列表[{'charge': z, 'conc_func': 浓度计算函数}]
        additional_cations: 额外阳离子浓度列表[(浓度, 电荷)]
        additional_anions: 额外阴离子浓度列表[(浓度, 电荷)]
    返回:
        电荷平衡残差
    """
    balance = H_conc - (Kw / H_conc)  # H+ - OH-
    
    # 添加额外阳离子贡献
    for conc, charge in additional_cations:
        balance += conc * charge
    
    # 添加额外阴离子贡献
    for conc, charge in additional_anions:
        balance -= conc * abs(charge)  # 阴离子电荷取绝对值
    
    # 添加其他物种贡献
    for spec in species:
        A_conc, HA_conc = spec['conc_func'](H_conc)  # 解包A⁻和HA浓度
        balance += A_conc * spec['charge_A']   # A⁻的电荷贡献
        balance += HA_conc * spec['charge_HA']  # HA的电荷贡献
    
    return balance

def calculate_ionic_strength(species, H_conc, additional_ions=[]):
    """
    通用离子强度计算
    参数:
        species: 物种列表[{'charge_A': zA, 'charge_HA': zHA, 'conc_func': 浓度计算函数}]
        H_conc: H+浓度
        additional_ions: 额外离子列表[(浓度, 电荷)]
    """
    I = 0.5 * (H_conc * 1**2 + (Kw / H_conc) * (-1)**2)  # H+和OH-贡献
    
    # 添加额外离子贡献
    for conc, charge in additional_ions:
        I += 0.5 * conc * charge**2
    
    # 添加其他物种贡献
    for spec in species:
        A_conc, HA_conc = spec['conc_func'](H_conc)  # 解包A⁻和HA浓度
        I += 0.5 * A_conc * spec['charge_A']**2  # A⁻的离子强度贡献
        I += 0.5 * HA_conc * spec['charge_HA']**2  # HA的离子强度贡献
    
    return I

def calculate_buffer_capacity(pH, buffers):
    """计算缓冲容量 β = 2.3 * Σ([HA][A⁻] / ([HA] + [A⁻]))"""
    H_conc = 10**(-pH)
    beta = 0.0
    for buf in buffers:
        Ka = 10**(-buf['pKa'])
        HA = buf['total_conc'] * H_conc / (H_conc + Ka)
        A = buf['total_conc'] - HA
        beta += 2.3 * HA * A / (HA + A) if (HA + A) > 0 else 0
    return beta

# ------------------------- 滴定曲线计算 -------------------------

def titration_curve(
    buffer_system,
    titrant_conc=0.1,
    titrant_charge=1,
    initial_volume=20.0,
    step_size=0.1
):
    """
        滴定曲线计算
    """
    data = {
        'Volume_mL': [],
        'pH': [],
        'Ionic_Strength': [],
        'Buffer_Capacity': []
    }

    # 初始总摩尔数
    initial_volume_L = initial_volume / 1000
    for spec in buffer_system['species']:
        spec['total_moles'] = spec['total_conc'] * initial_volume_L

    # 计算初始反离子浓度
    initial_cations, initial_anions = [], []
    if any(spec.get('ifaddition') == 1 for spec in buffer_system['species']):
        for spec in buffer_system['species']:
            if 'counter_ion_charge' in spec:
                charge = spec['counter_ion_charge']
                conc = spec['total_conc']
                (initial_cations if charge > 0 else initial_anions).append(
                    (conc, abs(charge))
                )

    # 稳健初值
    H_guess = 1e-7

    # 逐步滴定
    V_titrant = 0.0
    while True:
        current_volume_L = (initial_volume + V_titrant) / 1000
        moles_titrant = titrant_conc * V_titrant / 1000

        # 滴定剂引入的离子
        titrant_conc_current = moles_titrant / current_volume_L
        additional_cations = [(titrant_conc_current, titrant_charge)] \
            if titrant_charge > 0 else []
        additional_anions = [(titrant_conc_current, abs(titrant_charge))] \
            if titrant_charge < 0 else []

        # 稀释后的初始离子
        dil = initial_volume_L / current_volume_L
        current_cations = [(c * dil, z) for c, z in initial_cations] + additional_cations
        current_anions   = [(c * dil, z) for c, z in initial_anions] + additional_anions

        # 构造物种浓度函数 & buffers
        species_for_charge_balance, buffers_for_beta = [], []
        for spec in buffer_system['species']:
            if 'pKa' not in spec:
                continue

            def make_conc_func(sp):
                def conc_func(H_conc):
                    Ka = 10 ** (-sp['pKa'])
                    total = sp['total_moles'] / current_volume_L
                    if sp.get('initial_form') == 'A-':
                        HA = total * H_conc / (H_conc + Ka)
                        A  = total - HA
                    else:  # default HA
                        A  = total * Ka / (H_conc + Ka)
                        HA = total - A
                    return A, HA
                return conc_func

            species_for_charge_balance.append({
                'charge_A':  spec.get('charge_A', -1),
                'charge_HA': spec.get('charge_HA', 0),
                'conc_func': make_conc_func(spec)
            })
            buffers_for_beta.append({
                'pKa': spec['pKa'],
                'total_conc': spec['total_moles'] / current_volume_L
            })

        # 稳健求解电荷平衡
        def balance(H):
            return charge_balance(H, species_for_charge_balance,
                                  current_cations, current_anions)

        try:
            # 先用 brentq，区间 pH 0~14 => [1e-14, 1]
            H_conc = brentq(balance, 1e-14, 1, maxiter=200)
            current_pH = calculate_pH(H_conc)
            H_guess = max(min(H_conc, 1e-1), 1e-13)  # 更新初值
        except ValueError:
            # 区间同号，无解
            H_conc = np.nan
            current_pH = np.nan

        # 记录数据（即使 pH=NaN 也记录，保证连续性）
        data['Volume_mL'].append(V_titrant * titrant_charge)
        data['pH'].append(current_pH)

        # Ionic strength & buffer capacity
        if not np.isnan(H_conc):
            all_ions = current_cations + [(c, -z) for c, z in current_anions]
            I = calculate_ionic_strength(species_for_charge_balance,
                                         H_conc, all_ions)
            beta = calculate_buffer_capacity(current_pH, buffers_for_beta)
        else:
            I, beta = np.nan, np.nan

        data['Ionic_Strength'].append(I)
        data['Buffer_Capacity'].append(beta)

        # 终止条件：最大 50 mL
        V_titrant += step_size
        if V_titrant > 30:
            break
        if current_pH >12 or current_pH <2:
            break

    # 生成 DataFrame 并附加缓冲剂信息
    df = pd.DataFrame(data)
    for i, spec in enumerate(buffer_system['species']):
        if 'pKa' in spec:
            df[f'Component_{i+1}_Name'] = spec.get('name', f'Component_{i+1}')
            df[f'Component_{i+1}_pKa'] = spec['pKa']
            df[f'Component_{i+1}_Total_Conc_M'] = spec['total_conc']
            df[f'Component_{i+1}_Initial_Form'] = spec.get('initial_form', 'HA')
            if 'counter_ion_charge' in spec:
                df[f'Component_{i+1}_Counter_Ion_Charge'] = spec['counter_ion_charge']

    return df

# ------------------------- 示例调用 -------------------------
if __name__ == '__main__':
    # 示例1: 0.1M醋酸钠用0.1M HCl滴定
    """
    通用盐溶液滴定曲线计算
    参数:
        buffer_system: {
            'species': [
                {
                    'name': 'Acetate',
                    'pKa': 4.76,
                    'total_conc': 0.1,
                    'initial_form': 'A-',  # 'HA'或'A-',初始为共轭碱A-,酸HA
                    'charge_A': -2,           # CO₃²⁻电荷,电离的结果离子
                    'charge_HA': -1,          # HCO₃⁻电荷,作为电离的HA,酸溶液
                    'counter_ion_charge': 1  # 反离子电荷(如醋酸钠为1, Ca2+为2, 醋酸为1, 即不水解的那位)
                    'ifaddition': 1   #0:酸,例如醋酸,碳酸等未引入额外要计算的离子,1:碳酸氢钠等引入了Na+类似的离子。
                },
                # 可添加多个组分
            ]
        }
    """
    sodium_acetate = {
        'species': [
            {
                'name': 'Acetate',
                'pKa': 4.76,
                'total_conc': 0.05,
                'initial_form': 'A-',
                'charge_A': -1,           
                'charge_HA': 0,          
                'counter_ion_charge': 1,
                'ifaddition': 1
            }
        ]
    }
    
    df1 = titration_curve(
        sodium_acetate,
        titrant_conc=0.1,  #滴定计浓度
        titrant_charge=-1,  # HCL为-1，NaOH为1
        initial_volume=20.0,  #初始体积20mL
    )
    df1.to_csv('acetate_titration.csv', index=False)
    # KH2PO4
    kh2po4_system = {
    'species': [
        {
            'name': 'H2PO4-',
            'pKa': 7.21,  # H2PO4- ↔ HPO4²⁻ + H+
            'total_conc': 0.05,
            'initial_form': 'HA',  # 初始为 H2PO4⁻
            'charge_A': -2,
            'charge_HA': -1,
            'counter_ion_charge': 1,  # K⁺
            'ifaddition': 1
        }
    ]
    }
    
    df_kh2po4 = titration_curve(
        kh2po4_system,
        titrant_conc=0.1,
        titrant_charge=1,  # NaOH
        initial_volume=20.0,
    )
    df_kh2po4.to_csv('kh2po4_titration.csv', index=False)
    # NH4CL
    nh4cl_system = {
    'species': [
        {
            'name': 'NH4+',
            'pKa': 9.25,
            'total_conc': 0.05,
            'initial_form': 'HA',  # 初始为 NH4⁺
            'charge_A': 0,
            'charge_HA': 1,
            'counter_ion_charge': -1,
            'ifaddition': 1
        }
    ]
    }
    
    df_nh4cl = titration_curve(
        nh4cl_system,
        titrant_conc=0.1,
        titrant_charge=1,  # NaOH
        initial_volume=20.0,
    )
    df_nh4cl.to_csv('nh4cl_titration.csv', index=False)
    # Na₃C₆H₅O₇
    citrate_system = {
    'species': [
        {
            'name': 'Citrate_3-',
            'pKa': 6.40,  # HCit²⁻ ↔ Cit³⁻ + H+
            'total_conc': 0.05,
            'initial_form': 'A-',  # 初始为 Cit³⁻
            'charge_A': -3,
            'charge_HA': -2,
            'counter_ion_charge': 3,  # 3Na⁺
            'ifaddition': 1
        }
    ]
    }
    
    df_citrate = titration_curve(
        citrate_system,
        titrant_conc=0.1,
        titrant_charge=-1,
        initial_volume=20.0,
    )
    df_citrate.to_csv('citrate_titration.csv', index=False)
    #MIX
    mixed_buffer = {
    'species': [
        {
            'name': 'NH4+',
            'pKa': 9.25,
            'total_conc': 0.2,
            'initial_form': 'HA',  # 初始为 NH4⁺
            'charge_A': 0,
            'charge_HA': 1,
            'counter_ion_charge': -1,
            'ifaddition': 1
        },
        {
            'name': 'H2PO4-',
            'pKa': 7.21,  # H2PO4- ↔ HPO4²⁻ + H+
            'total_conc': 0.25,
            'initial_form': 'HA',  # 初始为 H2PO4⁻
            'charge_A': -2,
            'charge_HA': -1,
            'counter_ion_charge': 1,  # K⁺
            'ifaddition': 1
        }
    ]
    }
    
    df_mixed = titration_curve(
        mixed_buffer,
        titrant_conc=0.1,
        titrant_charge=1,  
        initial_volume=20.0,
    )
    df_mixed.to_csv('nh4cl_H2PO4_titration.csv', index=False)