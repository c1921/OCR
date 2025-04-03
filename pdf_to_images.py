from pdf2image import convert_from_path
import os
from tqdm import tqdm
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial

def pdf_to_images(pdf_path, output_dir, dpi=200, thread_count=4):
    """
    将PDF文件转换为图片
    
    参数:
        pdf_path (str): PDF文件的路径
        output_dir (str): 输出图片的目录
        dpi (int): 图片的分辨率，默认200
        thread_count (int): 转换使用的线程数，默认4
    """
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # 获取PDF文件名（不包含扩展名）
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # 为每个PDF创建单独的输出目录
        pdf_output_dir = os.path.join(output_dir, pdf_name)
        if not os.path.exists(pdf_output_dir):
            os.makedirs(pdf_output_dir)

        # 使用临时目录来避免内存溢出
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"\n正在处理PDF: {pdf_name}")
            
            # 使用pdftocairo和JPEG格式来提升性能
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                output_folder=temp_dir,
                fmt='jpeg',
                thread_count=thread_count,
                use_pdftocairo=True,
                output_file=pdf_name,
                paths_only=True  # 只返回文件路径而不是图片对象
            )
            
            total_pages = len(images)
            
            # 将临时文件移动到目标目录
            print("正在保存图片...")
            for i, image_path in enumerate(tqdm(images, desc="处理页面", unit="页")):
                output_file = os.path.join(pdf_output_dir, f"{pdf_name}_page_{i+1}.jpg")
                try:
                    # 使用文件复制替代图片对象操作
                    import shutil
                    shutil.copy2(image_path, output_file)
                except Exception as e:
                    print(f"保存第 {i+1} 页时出错: {str(e)}")
                    continue
        
        print(f"✓ PDF转换完成！共转换 {total_pages} 页")
        return True
        
    except Exception as e:
        print(f"转换过程中出现错误: {str(e)}")
        return False

def process_single_pdf(pdf_file, input_folder, output_folder, current, total):
    """处理单个PDF文件"""
    pdf_path = os.path.join(input_folder, pdf_file)
    print(f"\n[{current}/{total}] 处理PDF文件: {pdf_file}")
    return pdf_to_images(pdf_path, output_folder)

def process_pdf_folder(input_folder="./pdf_input", output_folder="./output_images", max_workers=2):
    """
    处理指定文件夹下的所有PDF文件
    
    参数:
        input_folder (str): 输入PDF文件夹路径
        output_folder (str): 输出图片文件夹路径
        max_workers (int): 同时处理的PDF文件数，默认2
    """
    # 确保输入文件夹存在
    if not os.path.exists(input_folder):
        print(f"输入文件夹 {input_folder} 不存在！")
        return False
    
    # 获取所有PDF文件
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"在 {input_folder} 中没有找到PDF文件！")
        return False
    
    total_pdfs = len(pdf_files)
    print(f"找到 {total_pdfs} 个PDF文件，开始处理...")
    
    # 使用线程池并行处理多个PDF文件
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, pdf_file in enumerate(pdf_files):
            # 直接创建future对象
            future = executor.submit(
                process_single_pdf,
                pdf_file,
                input_folder,
                output_folder,
                i + 1,
                total_pdfs
            )
            futures.append(future)
        
        # 等待所有任务完成并获取结果
        results = [f.result() for f in futures]
    
    # 统计处理结果
    success_count = sum(1 for r in results if r)
    print(f"\n✓ 所有PDF文件处理完成！成功: {success_count}/{total_pdfs}")
    return True

if __name__ == "__main__":
    # 处理pdf_input文件夹下的所有PDF文件
    process_pdf_folder() 