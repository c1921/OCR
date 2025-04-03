from pdf2image import convert_from_path
import os

def pdf_to_images(pdf_path, output_dir, dpi=200):
    """
    将PDF文件转换为图片
    
    参数:
        pdf_path (str): PDF文件的路径
        output_dir (str): 输出图片的目录
        dpi (int): 图片的分辨率，默认200
    """
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # 将PDF转换为图片
        images = convert_from_path(pdf_path, dpi=dpi)
        
        # 获取PDF文件名（不包含扩展名）
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # 为每个PDF创建单独的输出目录
        pdf_output_dir = os.path.join(output_dir, pdf_name)
        if not os.path.exists(pdf_output_dir):
            os.makedirs(pdf_output_dir)
        
        # 保存每一页为单独的图片
        for i, image in enumerate(images):
            # 生成输出文件名
            output_file = os.path.join(pdf_output_dir, f"{pdf_name}_page_{i+1}.png")
            # 保存图片
            image.save(output_file, "PNG")
            print(f"已保存第 {i+1} 页: {output_file}")
            
        print(f"转换完成！共转换 {len(images)} 页")
        return True
        
    except Exception as e:
        print(f"转换过程中出现错误: {str(e)}")
        return False

def process_pdf_folder(input_folder="./pdf_input", output_folder="./output_images"):
    """
    处理指定文件夹下的所有PDF文件
    
    参数:
        input_folder (str): 输入PDF文件夹路径
        output_folder (str): 输出图片文件夹路径
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
    
    print(f"找到 {len(pdf_files)} 个PDF文件，开始处理...")
    
    # 处理每个PDF文件
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_file)
        print(f"\n正在处理: {pdf_file}")
        pdf_to_images(pdf_path, output_folder)
    
    print("\n所有PDF文件处理完成！")
    return True

if __name__ == "__main__":
    # 处理pdf_input文件夹下的所有PDF文件
    process_pdf_folder() 