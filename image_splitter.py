import cv2
import numpy as np
from collections import Counter
import os

def detect_paragraph_starts(image_path):
    # 读取图像
    img = cv2.imread(image_path)
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

def split_image_by_paragraphs(image_path, output_dir="output"):
    """
    根据段落检测结果裁切图像并保存
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取图像
    img = cv2.imread(image_path)
    text_lines, indented_lines = detect_paragraph_starts(image_path)
    
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
        # 裁切段落
        para_img = img[start_y:end_y, :]
        
        # 生成输出文件名
        output_path = os.path.join(output_dir, f'paragraph_{i+1}.png')
        
        # 保存图像
        cv2.imwrite(output_path, para_img)

def visualize_paragraphs(image_path):
    img = cv2.imread(image_path)
    text_lines, indented_lines = detect_paragraph_starts(image_path)
    
    # 在图像上标记段落起始
    for i, (start_y, end_y) in enumerate(text_lines):
        if i in indented_lines:
            # 在新段落处画一个红色标记
            cv2.line(img, (0, start_y-2), (50, start_y-2), (0, 0, 255), 2)
    
    # 显示结果
    cv2.imshow('Paragraph Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用示例
if __name__ == "__main__":
    image_path = "files/test.png"
    # 可视化段落检测结果
    visualize_paragraphs(image_path)
    
    # 裁切图像并保存段落
    split_image_by_paragraphs(image_path, "output_paragraphs")
