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

# Set font for matplotlib based on OS to support various characters
font_settings = {
    "Windows": "SimHei",       # Using SimHei on Windows for broad character support
    "Darwin": "Arial Unicode MS",  # Mac
    "Linux": "WenQuanYi Zen Hei" # A common choice for Linux
}
system = platform.system()
rcParams['font.sans-serif'] = [font_settings.get(system, 'SimHei')]
rcParams['axes.unicode_minus'] = False


def is_contaminated(img,
                    channel_diff_threshold=5,
                    pixel_ratio_threshold=0.001,
                    edge_mask_radius=0):
    """
    Adaptive contamination detection function.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Edge masking process
    if edge_mask_radius > 0:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask,
                      (edge_mask_radius, edge_mask_radius),
                      (w - edge_mask_radius, h - edge_mask_radius),
                      255, -1)
        img_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

    # Calculate channel difference
    max_val = np.max(img_rgb, axis=2)
    min_val = np.min(img_rgb, axis=2)
    diff = max_val - min_val

    contaminated_pixels = np.sum(diff > channel_diff_threshold)
    total_pixels = h * w
    return contaminated_pixels > (total_pixels * pixel_ratio_threshold), contaminated_pixels


def generate_contamination_report(contaminated_data, input_dir, output_dir):
    """
    Generate a visual report.
    """
    report_dir = os.path.join(output_dir, "contamination_reports")
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Calculate statistics
    total_images = len(contaminated_data) + len(
        [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    stats = {
        "Detection Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Total Images Scanned": total_images,
        "Contaminated Images Found": len(contaminated_data),
        "Contamination Ratio (%)": round(len(contaminated_data) / total_images * 100, 2) if total_images > 0 else 0,
        "Average Contaminated Pixels": int(np.mean([d['contaminated_pixels'] for d in contaminated_data])) if contaminated_data else 0,
        "Max Contaminated Pixels": max([d['contaminated_pixels'] for d in contaminated_data]) if contaminated_data else 0,
        "Channel Difference Threshold": contaminated_data[0]['channel_diff_threshold'] if contaminated_data else 0
    }

    # Create a DataFrame for the report
    df = pd.DataFrame([{
        "Filename": d['filename'],
        "Image Dimensions": f"{d['width']}x{d['height']}",
        "Total Pixels": d['total_pixels'],
        "Contaminated Pixels": d['contaminated_pixels'],
        "Contamination Ratio (%)": round(d['contaminated_pixels'] / d['total_pixels'] * 100, 3),
        "Processing Time (s)": d['processing_time'],
        "Detection Time": d['detection_time']
    } for d in contaminated_data])

    # Generate histogram
    plt.figure(figsize=(12, 6))
    plt.hist(df["Contamination Ratio (%)"], bins=20, edgecolor='black')
    plt.title("Distribution of Contaminated Pixel Ratios")
    plt.xlabel("Contamination Ratio (%)")
    plt.ylabel("Number of Images")
    hist_path = os.path.join(report_dir, f"contamination_hist_{timestamp}.png")
    plt.savefig(hist_path)
    plt.close()

    # Generate HTML report
    report_path = os.path.join(report_dir, f"contamination_report_{timestamp}.html")
    html_content = f"""
    <html>
    <head>
        <title>Contamination Detection Report - {timestamp}</title>
        <style>
            body {{ font-family: sans-serif; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            h1, h2 {{ color: #333; }}
        </style>
    </head>
    <body>
        <h1>Medical Image Contamination Detection Report</h1>
        <h2>Overall Statistics</h2>
        <ul>
            {"".join([f"<li><b>{k}</b>: {v}</li>" for k, v in stats.items()])}
        </ul>
        <h2>Contamination Distribution</h2>
        <img src="{os.path.basename(hist_path)}" alt="Contamination Distribution Histogram" width="800">
        <h2>Detailed Data</h2>
        {df.sort_values('Contaminated Pixels', ascending=False).to_html(index=False)}
    </body>
    </html>
    """

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return report_path


def process_subdirectory(input_dir, output_root_dir):
    """
    Complete processing pipeline for a single subdirectory.
    """
    dir_name = os.path.basename(input_dir)
    output_dir = os.path.join(output_root_dir, f"{dir_name}_contaminated")

    print(f"\n{'=' * 40}")
    print(f"Processing directory: {dir_name}")

    # Run contamination detection
    auto_detect_contaminated_batch(
        input_dir=input_dir,
        output_contaminated_dir=output_dir
    )

    print(f"Finished processing: {dir_name}")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 40}\n")


def auto_detect_contaminated_batch(
        input_dir,
        output_contaminated_dir,
        channel_diff_threshold=12,
        pixel_ratio_threshold=0.0005,
        edge_mask_radius=20
):
    """
    Improved batch detection function that processes all images in a directory.
    """
    os.makedirs(output_contaminated_dir, exist_ok=True)
    img_files = [f for f in os.listdir(input_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Automatic threshold adjustment (example code, enable and customize as needed)
    # if len(img_files) > 10:
    #     thresholds = []
    #     for f in img_files[:10]:
    #         img = cv2.imread(os.path.join(input_dir, f))
    #         if img is None: continue
    #         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         diff = np.max(img_rgb, axis=2) - np.min(img_rgb, axis=2)
    #         thresholds.append(np.median(diff))
    #     if thresholds:
    #         channel_diff_threshold = max(channel_diff_threshold, int(np.mean(thresholds)) + 5)

    contaminated_data = []
    for filename in tqdm(img_files, desc=f"Processing {os.path.basename(input_dir)}", leave=False):
        img_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_contaminated_dir, filename)

        try:
            start_time = datetime.now()
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {filename}. Skipping.")
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
            print(f"Error processing {filename}: {str(e)}")

    # Generate a report if contaminated files were found
    if contaminated_data:
        generate_contamination_report(contaminated_data, input_dir, output_contaminated_dir)


def batch_process_main_directory(main_input_dir, main_output_dir):
    """
    Main entry function for batch processing a directory containing subdirectories.
    """
    if not os.path.exists(main_input_dir):
        raise FileNotFoundError(f"Input directory not found: {main_input_dir}")

    sub_dirs = [d for d in os.listdir(main_input_dir)
                if os.path.isdir(os.path.join(main_input_dir, d))]

    if not sub_dirs:
        print("No subdirectories found to process. Processing the main directory instead.")
        process_subdirectory(main_input_dir, main_output_dir)
        return

    print(f"\nStarting batch process. Found {len(sub_dirs)} subdirectories.")
    print(f"Main input directory: {main_input_dir}")
    print(f"Main output directory: {main_output_dir}")

    os.makedirs(main_output_dir, exist_ok=True)

    for sub_dir in tqdm(sub_dirs, desc="Overall Progress"):
        input_subdir_path = os.path.join(main_input_dir, sub_dir)
        try:
            process_subdirectory(input_subdir_path, main_output_dir)
        except Exception as e:
            print(f"Failed to process {sub_dir}: {str(e)}")
            continue

    print("\nProcessing complete! All results have been saved to:")
    print(f"→ {os.path.abspath(main_output_dir)}")


if __name__ == "__main__":
    # --- Usage Example ---
    # Main input directory containing one or more subdirectories with images
    input_main_dir = r"C:\Users\YourUser\Desktop\image_source_folder"
    # Main output directory where results will be saved
    output_main_dir = r"C:\Users\YourUser\Desktop\processed_images"

    batch_process_main_directory(
        main_input_dir=input_main_dir,
        main_output_dir=output_main_dir
    )