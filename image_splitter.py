import cv2
import numpy as np
import os
import glob

def detect_paragraph_starts(image_path):
    # 读取图像 - 使用 imdecode
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return [], []
        
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 使用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 使用Otsu's二值化
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 获取水平投影
    h_proj = np.sum(binary, axis=1)
    
    # 找到文本行的位置
    text_lines = []
    line_spaces = []
    last_end = None
    start = None
    min_line_height = 10
    
    # 使用动态阈值找到文本行
    threshold = np.max(h_proj) * 0.05
    
    for i, proj in enumerate(h_proj):
        if proj > threshold and start is None:
            start = i
        elif (proj <= threshold or i == len(h_proj)-1) and start is not None:
            if i - start >= min_line_height:
                text_lines.append((start, i))
                if last_end is not None:
                    line_spaces.append(start - last_end)
                last_end = i
            start = None
    
    # 分析行间距
    if line_spaces:
        avg_space = np.mean(line_spaces)
        std_space = np.std(line_spaces)
    
    # 对每一行检测最左边的文本位置
    paragraph_starts = []
    for start_y, end_y in text_lines:
        # 扩展行高以确保捕获完整字符
        start_y = max(0, start_y - 2)
        end_y = min(binary.shape[0], end_y + 2)
        
        line_img = binary[start_y:end_y, :]
        # 获取垂直投影
        v_proj = np.sum(line_img, axis=0)
        
        # 使用滑动窗口平滑处理
        window_size = 10
        smoothed_proj = np.convolve(v_proj, np.ones(window_size)/window_size, mode='valid')
        
        # 找到第一个显著文本位置
        text_threshold = np.max(smoothed_proj) * 0.15
        for x, proj in enumerate(smoothed_proj):
            if proj > text_threshold:
                paragraph_starts.append(x)
                break
    
    # 分析段落起始位置
    indented_lines = []
    if paragraph_starts:
        # 计算起始位置的统计信息
        starts_array = np.array(paragraph_starts)
        median_start = np.median(starts_array)
        std_dev = np.std(starts_array)
        
        # 第一行总是新段落
        indented_lines.append(0)
        
        for i in range(1, len(paragraph_starts)):
            is_new_paragraph = False
            
            # 检查缩进（相对于中位数）
            if paragraph_starts[i] > median_start + std_dev:
                is_new_paragraph = True
            
            # 检查行间距（如果大于平均行间距的1.5倍）
            if i > 0 and line_spaces[i-1] > avg_space * 1.5:
                is_new_paragraph = True
                
            if is_new_paragraph:
                indented_lines.append(i)
    
    return text_lines, indented_lines

def split_image_by_paragraphs(image_path, output_dir="output_paragraphs", marked_dir="output_marked"):
    """
    根据段落检测结果裁切图像并保存
    
    Args:
        image_path: 输入图像路径
        output_dir: 裁切后段落的输出目录路径
        marked_dir: 标记后图像的输出目录路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(marked_dir, exist_ok=True)
    
    # 读取图像 - 使用完整的 Unicode 路径
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 获取检测结果
    text_lines, indented_lines = detect_paragraph_starts(image_path)
    
    # 创建标记后的图像副本
    marked_img = img.copy()
    
    # 在图像上标记段落起始
    for i, (start_y, end_y) in enumerate(text_lines):
        if i in indented_lines:
            cv2.line(marked_img, (0, start_y-2), (50, start_y-2), (0, 0, 255), 2)
    
    # 保存标记后的图像
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    marked_path = os.path.join(marked_dir, f'{base_name}_marked.png')
    # 使用 imencode 保存图像
    cv2.imencode('.png', marked_img)[1].tofile(marked_path)
    
    # 获取段落的起始和结束位置
    paragraph_bounds = []
    start_idx = 0
    
    for i in range(len(text_lines)):
        if i + 1 in indented_lines or i + 1 == len(text_lines):
            # 获取段落的开始和结束行
            para_start = text_lines[start_idx][0]
            para_end = text_lines[i][1]
            
            # 添加额外的边距
            margin = 5
            para_start = max(0, para_start - margin)
            para_end = min(img.shape[0], para_end + margin)
            
            paragraph_bounds.append((para_start, para_end))
            start_idx = i + 1
    
    # 裁切并保存每个段落
    for i, (start_y, end_y) in enumerate(paragraph_bounds):
        para_img = img[start_y:end_y, :]
        output_path = os.path.join(output_dir, f'{base_name}_para_{i+1}.png')
        # 使用 imencode 保存图像
        cv2.imencode('.png', para_img)[1].tofile(output_path)

def process_directory(input_dir="img_input", output_dir="output_paragraphs", marked_dir="output_marked"):
    """
    处理输入目录中的所有图像
    
    Args:
        input_dir: 输入图像目录
        output_dir: 裁切后段落的输出目录
        marked_dir: 标记后图像的输出目录
    """
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        print(f"错误：输入目录 {input_dir} 不存在！")
        return
    
    # 获取所有图像文件
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_files:
        print(f"错误：在 {input_dir} 目录下没有找到图像文件！")
        return
    
    print(f"==> 开始处理图像...")
    
    # 处理每个图像文件
    for image_path in image_files:
        print(f"正在处理: {os.path.basename(image_path)}")
        split_image_by_paragraphs(image_path, output_dir, marked_dir)
    
    print(f"==> 处理完成！")
    print(f"裁切后的段落保存在: {output_dir}")
    print(f"标记后的图像保存在: {marked_dir}")

if __name__ == "__main__":
    process_directory()
