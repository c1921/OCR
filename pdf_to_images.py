from pdf2image import convert_from_path
import os
from tqdm import tqdm
import tempfile
from concurrent.futures import ThreadPoolExecutor
import yaml

def load_config():
    """加载配置文件"""
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config['pdf_settings']
    except Exception as e:
        print(f"加载配置文件失败: {str(e)}")
        # 返回默认配置
        return {
            'input_folder': './pdf_input',
            'output_folder': './output_images',
            'dpi': 96,
            'thread_count': 8,
            'max_workers': 2,
            'format': 'jpeg',
            'quality': 95
        }

def pdf_to_images(pdf_path, output_dir, config):
    """
    将PDF文件转换为图片
    
    参数:
        pdf_path (str): PDF文件的路径
        output_dir (str): 输出图片的目录
        config (dict): 配置参数
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
            
            # 使用pdftocairo和配置的格式来转换
            images = convert_from_path(
                pdf_path,
                dpi=config['dpi'],
                output_folder=temp_dir,
                fmt=config['format'],
                thread_count=config['thread_count'],
                use_pdftocairo=True,
                output_file=pdf_name,
                paths_only=True
            )
            
            total_pages = len(images)
            
            # 将临时文件移动到目标目录
            print("正在保存图片...")
            for i, image_path in enumerate(tqdm(images, desc="处理页面", unit="页")):
                output_file = os.path.join(pdf_output_dir, f"{pdf_name}_page_{i+1}.{config['format']}")
                try:
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

def process_single_pdf(pdf_file, input_folder, output_folder, current, total, config):
    """处理单个PDF文件"""
    pdf_path = os.path.join(input_folder, pdf_file)
    print(f"\n[{current}/{total}] 处理PDF文件: {pdf_file}")
    return pdf_to_images(pdf_path, output_folder, config)

def process_pdf_folder():
    """处理指定文件夹下的所有PDF文件"""
    # 加载配置
    config = load_config()
    input_folder = config['input_folder']
    output_folder = config['output_folder']
    max_workers = config['max_workers']
    
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
            future = executor.submit(
                process_single_pdf,
                pdf_file,
                input_folder,
                output_folder,
                i + 1,
                total_pdfs,
                config
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