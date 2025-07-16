import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import platform
from matplotlib import rcParams

# 设置中文字体
plt_fonts = {
    "Windows": "SimHei",
    "Darwin": "Arial Unicode MS",  # Mac
    "Linux": "WenQuanYi Zen Hei"
}
system = platform.system()
rcParams['font.sans-serif'] = [plt_fonts.get(system, 'SimHei')]
rcParams['axes.unicode_minus'] = False


def is_contaminated(img,
                    channel_diff_threshold=5,
                    pixel_ratio_threshold=0.001,
                    edge_mask_radius=0):
    """
    自适应污染检测函数
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # 边缘屏蔽处理
    if edge_mask_radius > 0:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask,
                      (edge_mask_radius, edge_mask_radius),
                      (w - edge_mask_radius, h - edge_mask_radius),
                      255, -1)
        img_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

    # 计算通道差异
    max_val = np.max(img_rgb, axis=2)
    min_val = np.min(img_rgb, axis=2)
    diff = max_val - min_val

    contaminated_pixels = np.sum(diff > channel_diff_threshold)
    total_pixels = h * w
    return contaminated_pixels > (total_pixels * pixel_ratio_threshold), contaminated_pixels


def generate_contamination_report(contaminated_data, input_dir, output_dir):
    """
    生成带中文的可视化报告
    """
    report_dir = os.path.join(output_dir, "contamination_reports")
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 统计信息计算
    total_images = len(contaminated_data) + len(
        [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    stats = {
        "检测时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "总检测图片数": total_images,
        "污染图片数": len(contaminated_data),
        "污染比例 (%)": round(len(contaminated_data) / total_images * 100, 2),
        "平均污染像素数": int(np.mean([d['contaminated_pixels'] for d in contaminated_data])),
        "最大污染像素数": max([d['contaminated_pixels'] for d in contaminated_data]),
        "污染像素阈值": contaminated_data[0]['channel_diff_threshold'] if contaminated_data else 0
    }

    # 创建数据表格
    df = pd.DataFrame([{
        "文件名": d['filename'],
        "图像尺寸": f"{d['width']}x{d['height']}",
        "总像素数": d['total_pixels'],
        "污染像素数": d['contaminated_pixels'],
        "污染比例 (%)": round(d['contaminated_pixels'] / d['total_pixels'] * 100, 3),
        "检测耗时 (s)": d['processing_time'],
        "检测时间": d['detection_time']
    } for d in contaminated_data])

    # 生成直方图
    plt.figure(figsize=(12, 6))
    plt.hist(df["污染比例 (%)"], bins=20, edgecolor='black')
    plt.title("污染像素比例分布直方图")
    plt.xlabel("污染比例 (%)")
    plt.ylabel("图片数量")
    hist_path = os.path.join(report_dir, f"contamination_hist_{timestamp}.png")
    plt.savefig(hist_path)
    plt.close()

    # 生成HTML报告
    report_path = os.path.join(report_dir, f"contamination_report_{timestamp}.html")
    html_content = f"""
    <html>
    <head>
        <title>污染检测报告 - {timestamp}</title>
        <style>
            table {{border-collapse: collapse; width: 100%;}}
            th, td {{border: 1px solid #ddd; padding: 8px;}}
            tr:nth-child(even){{background-color: #f2f2f2;}}
        </style>
    </head>
    <body>
        <h1>医学影像污染检测报告</h1>
        <h2>总体统计</h2>
        <ul>
            {"".join([f"<li><b>{k}</b>: {v}</li>" for k, v in stats.items()])}
        </ul>
        <h2>污染分布</h2>
        <img src="{os.path.basename(hist_path)}" width="800">
        <h2>详细数据</h2>
        {df.sort_values('污染像素数', ascending=False).to_html(index=False)}
    </body>
    </html>
    """

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return report_path


def process_subdirectory(input_dir, output_root_dir):
    """
    处理单个子目录的完整流程
    """
    dir_name = os.path.basename(input_dir)
    output_dir = os.path.join(output_root_dir, f"{dir_name}_contaminated")

    print(f"\n{'=' * 40}")
    print(f"开始处理目录: {dir_name}")

    # 执行污染检测
    auto_detect_contaminated_batch(
        input_dir=input_dir,
        output_contaminated_dir=output_dir
    )

    print(f"完成处理: {dir_name}")
    print(f"结果保存在: {output_dir}")
    print(f"{'=' * 40}\n")


def auto_detect_contaminated_batch(
        input_dir,
        output_contaminated_dir,
        channel_diff_threshold=12,
        pixel_ratio_threshold=0.0005,
        edge_mask_radius=20
):
    """
    改进的批量检测函数（支持子目录处理）
    """
    os.makedirs(output_contaminated_dir, exist_ok=True)
    img_files = [f for f in os.listdir(input_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 自动阈值调整（示例代码，可根据需求启用）
    # if len(img_files) > 10:
    #     thresholds = []
    #     for f in img_files[:10]:
    #         img = cv2.imread(os.path.join(input_dir, f))
    #         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         diff = np.max(img_rgb, axis=2) - np.min(img_rgb, axis=2)
    #         thresholds.append(np.median(diff))
    #     channel_diff_threshold = max(channel_diff_threshold, int(np.mean(thresholds)) + 5)

    contaminated_data = []
    for filename in tqdm(img_files, desc=f"处理 {os.path.basename(input_dir)}", leave=False):
        img_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_contaminated_dir, filename)

        try:
            start_time = datetime.now()
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            is_contam, count = is_contaminated(
                img,
                channel_diff_threshold=channel_diff_threshold,
                pixel_ratio_threshold=pixel_ratio_threshold,
                edge_mask_radius=edge_mask_radius
            )

            if is_contam:
                shutil.move(img_path, output_path)
                contaminated_data.append({
                    "filename": filename,
                    "width": w,
                    "height": h,
                    "total_pixels": h * w,
                    "contaminated_pixels": count,
                    "channel_diff_threshold": channel_diff_threshold,
                    "processing_time": round((datetime.now() - start_time).total_seconds(), 2),
                    "detection_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

        except Exception as e:
            print(f"处理 {filename} 出错: {str(e)}")

    # 生成报告
    if contaminated_data:
        generate_contamination_report(contaminated_data, input_dir, output_contaminated_dir)


def batch_process_main_directory(main_input_dir, main_output_dir):
    """
    主目录批量处理入口函数
    """
    if not os.path.exists(main_input_dir):
        raise FileNotFoundError(f"输入目录不存在: {main_input_dir}")

    sub_dirs = [d for d in os.listdir(main_input_dir)
                if os.path.isdir(os.path.join(main_input_dir, d))]

    print(f"\n开始批量处理，共发现 {len(sub_dirs)} 个子目录")
    print(f"输入主目录: {main_input_dir}")
    print(f"输出主目录: {main_output_dir}")

    os.makedirs(main_output_dir, exist_ok=True)

    for idx, sub_dir in enumerate(tqdm(sub_dirs, desc="总进度"), 1):
        input_subdir = os.path.join(main_input_dir, sub_dir)
        try:
            process_subdirectory(input_subdir, main_output_dir)
        except Exception as e:
            print(f"处理 {sub_dir} 失败: {str(e)}")
            continue

    print("\n处理完成！所有结果已保存在:")
    print(f"→ {os.path.abspath(main_output_dir)}")


if __name__ == "__main__":
    # 使用示例
    input_main = r"C:\Users\20253\Desktop\ground_truth"  # 包含多个子目录的主输入目录
    output_main = r"C:\Users\20253\Desktop\hip_gen_samples"  # 主输出目录

    batch_process_main_directory(
        main_input_dir=input_main,
        main_output_dir=output_main
    )