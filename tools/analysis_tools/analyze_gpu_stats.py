import pandas as pd
import sys
import os

def analyze_gpu_csv(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 错误: 文件不存在 -> {file_path}")
        sys.exit(1)

    # 读取CSV
    df = pd.read_csv(file_path)

    # 清理列名
    df.columns = df.columns.str.strip()
    print(f"✅ 读取到的列名: {list(df.columns)}")

    expected_cols = ['temperature.gpu', 'utilization.gpu [%]', 'memory.used [MiB]', 'power.draw [W]']
    for col in expected_cols:
        if col not in df.columns:
            print(f"❌ 错误: 缺少列名 {col}，请检查CSV文件表头")
            sys.exit(1)

    # 去掉单位并转为数值
    df['temperature.gpu'] = df['temperature.gpu'].astype(float)
    df['utilization.gpu [%]'] = df['utilization.gpu [%]'].str.replace('%', '').astype(float)
    df['memory.used [MiB]'] = df['memory.used [MiB]'].str.replace('MiB', '').astype(float)
    df['power.draw [W]'] = df['power.draw [W]'].str.replace('W', '').astype(float)

    # 定义指标
    columns = {
        'temperature.gpu': 'GPU温度 (°C)',
        'utilization.gpu [%]': 'GPU利用率 (%)',
        'memory.used [MiB]': '显存使用量 (MiB)',
        'power.draw [W]': '功耗 (W)'
    }

    stats = {}

    print("\n====== GPU 性能统计结果 ======\n")

    for col, name in columns.items():
        avg = df[col].mean()
        maxv = df[col].max()
        minv = df[col].min()
        stdv = df[col].std()
        stats[col] = (avg, maxv, minv, stdv)

        print(f"{name}:")
        print(f"  平均值: {avg:.2f}")
        print(f"  最大值: {maxv:.2f}")
        print(f"  最小值: {minv:.2f}")
        print(f"  标准差: {stdv:.2f}\n")

    # === 自动生成简要总结 ===
    temp_avg, _, _, _ = stats['temperature.gpu']
    util_avg, util_max, util_min, util_std = stats['utilization.gpu [%]']
    power_avg, _, _, _ = stats['power.draw [W]']
    mem_avg, _, _, _ = stats['memory.used [MiB]']

    print("====== 简要总结 ======\n")

    print(
        f"GPU 平均温度为 {temp_avg:.1f}°C，"
        f"平均功耗约 {power_avg:.1f}W，"
        f"显存使用稳定在 {mem_avg:.0f} MiB。"
    )

    if util_avg > 80:
        load_desc = "GPU 长时间高负载运行"
    elif util_avg > 50:
        load_desc = "GPU 处于中等负载"
    else:
        load_desc = "GPU 利用率较低"

    print(
        f"整体利用率均值为 {util_avg:.1f}%（范围 {util_min:.0f}%-{util_max:.0f}%，波动标准差 {util_std:.1f}），"
        f"说明 {load_desc}。\n"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python analyze_gpu_stats.py <csv文件路径>")
        sys.exit(1)

    file_path = sys.argv[1]
    analyze_gpu_csv(file_path)