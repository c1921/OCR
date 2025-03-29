from paddleocr import PaddleOCR
import os
import glob

def process_image(img_path, ocr_model):
    """处理单个图片的OCR识别"""
    # 使用 PaddleOCR 进行文字识别
    result = ocr_model.ocr(img_path, cls=True)
    
    # 解析识别结果，将所有文本拼接成一行
    text_result = ""
    for idx, res in enumerate(result):
        for line in res:
            text_result += line[1][0]
    
    return text_result

def main():
    # 初始化 PaddleOCR 模型
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", det_db_box_thresh=0.3)
    
    # 指定输入目录（段落图片所在目录）
    input_dir = "./output_paragraphs"
    
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        print(f"错误：输入目录 {input_dir} 不存在！")
        exit(1)
    
    # 获取所有段落图片
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        # 修改文件匹配模式为 *_para_*.png
        image_files.extend(glob.glob(os.path.join(input_dir, f"*_para_*{ext}")))
    
    # 对文件名进行自然排序
    image_files = sorted(image_files, key=lambda x: (
        os.path.basename(x).split('_para_')[0],  # 首先按基础文件名排序
        int(os.path.basename(x).split('_para_')[1].split('.')[0])  # 然后按段落序号排序
    ))
    
    if not image_files:
        print(f"错误：在 {input_dir} 目录下没有找到段落图片！")
        exit(1)
    
    # 指定输出目录
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个段落图片并将结果写入同一个文件
    output_filename = os.path.join(output_dir, "paragraphs_ocr.txt")
    
    print("==> 开始OCR识别...")
    
    with open(output_filename, "w", encoding="utf-8") as f:
        current_base_name = None
        
        for img_path in image_files:
            base_name = os.path.basename(img_path).split('_para_')[0]
            
            # 如果是新的文件，添加文件名作为标题
            if base_name != current_base_name:
                if current_base_name is not None:
                    f.write("\n\n" + "="*50 + "\n\n")  # 文件之间的分隔符
                f.write(f"# {base_name}\n\n")
                current_base_name = base_name
            
            print(f"正在处理: {os.path.basename(img_path)}")
            
            # OCR识别
            text_result = process_image(img_path, ocr)
            
            # 写入结果，添加段落分隔符
            f.write(text_result + "\n\n")  # 段落之间用两个换行分隔
            
            # 同时输出到终端
            print("识别结果:")
            print(text_result)
            print()
    
    print(f"==> OCR识别完成，所有结果已写入 {output_filename}")

if __name__ == "__main__":
    main()
