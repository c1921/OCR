import cv2
import numpy as np
import os
import glob
from config import *

def detect_text_lines(img):
    """检测图像中的文本行
    
    Args:
        img: 输入的灰度图像
    
    Returns:
        lines: 文本行的y坐标列表，每个元素为(y_start, y_end)
    """
    # 二值化
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 使用形态学操作合并同一行的文本
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 获取每行文本的边界
    lines = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 过滤掉太窄或太高的区域
        if w > img.shape[1] * 0.1 and h < img.shape[0] * 0.1:  
            lines.append((y, y + h))
    
    # 按y坐标排序
    return sorted(lines, key=lambda x: x[0])

def detect_large_gaps(image_path, min_gap=70, max_gap=100):
    """检测文本行之间的大间距
    
    Args:
        image_path: 图像文件路径
        min_gap: 最小间距阈值（像素）
        max_gap: 最大间距阈值（像素）
    
    Returns:
        gaps: 大间距的位置列表，每个元素为(y_position, gap_size, is_extra_large)
    """
    # 读取图像
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return []
    
    # 检测文本行
    lines = detect_text_lines(img)
    
    # 检测间距
    gaps = []
    for i in range(len(lines) - 1):
        current_line_end = lines[i][1]
        next_line_start = lines[i + 1][0]
        gap_size = next_line_start - current_line_end
        
        if gap_size >= min_gap:
            # 添加标志表示是否为超大间距
            is_extra_large = gap_size >= max_gap
            gaps.append((current_line_end, gap_size, is_extra_large))
    
    return gaps

def find_optimal_cut_position(img, y_pos, gap_size, margin=5):
    """在间距区域内找到最佳切割位置（中间位置）
    
    Args:
        img: 灰度图像
        y_pos: 间距起始位置
        gap_size: 间距大小
        margin: 搜索边距
        
    Returns:
        optimal_y: 最佳切割位置
    """
    # 直接返回间距中间位置
    return y_pos + gap_size // 2

def split_by_gaps(img, gaps):
    """根据间距切割图像
    
    Args:
        img: 输入图像
        gaps: 间距列表，每个元素为(y_position, gap_size, is_extra_large)
    
    Returns:
        sections: 切割后的图像段落列表
        cut_positions: 实际切割位置列表
    """
    sections = []
    cut_positions = []
    height = img.shape[0]
    
    # 转换为灰度图用于查找最佳切割位置
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 处理第一段（从开始到第一个切割点）
    if gaps:
        first_cut = find_optimal_cut_position(gray, gaps[0][0], gaps[0][1])
        if first_cut > 0:  # 确保有内容才添加
            sections.append(img[0:first_cut])
        cut_positions.append(first_cut)
    
    # 处理中间的段落
    for i in range(len(gaps)):
        current_cut = cut_positions[i]
        
        # 如果是最后一个间距
        if i == len(gaps) - 1:
            # 添加最后一段
            if current_cut < height:  # 确保有内容才添加
                sections.append(img[current_cut:height])
        else:
            # 获取下一个切割点
            next_cut = find_optimal_cut_position(gray, gaps[i+1][0], gaps[i+1][1])
            cut_positions.append(next_cut)
            
            # 添加当前切割点到下一个切割点之间的段落
            if next_cut > current_cut:  # 确保有内容才添加
                sections.append(img[current_cut:next_cut])
    
    # 如果没有间距，则保留整个图像
    if not gaps:
        sections.append(img)
    
    # 验证切割结果
    total_height = sum(section.shape[0] for section in sections)
    if total_height != height:
        print(f"警告：切割后的总高度 ({total_height}) 与原图高度 ({height}) 不匹配！")
    
    return sections, cut_positions

def mark_gaps(image_path, gaps, cut_positions=None):
    """在图像上标记大间距
    
    Args:
        image_path: 图像文件路径
        gaps: 间距列表，每个元素为(y_position, gap_size, is_extra_large)
        cut_positions: 实际切割位置列表
    
    Returns:
        marked_img: 标记后的图像
    """
    # 读取图像
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    # 标记间距
    for i, (y_pos, gap_size, is_extra_large) in enumerate(gaps):
        # 根据间距大小选择颜色
        color = (255, 0, 0) if is_extra_large else (0, 0, 255)  # 蓝色或红色
        
        # 画虚线标记间距范围（通过绘制多个小线段实现）
        width = img.shape[1]
        dash_length = 10
        gap_length = 5
        x = 0
        while x < width:
            x_end = min(x + dash_length, width)
            # 绘制间距上边界
            cv2.line(img, (x, y_pos), (x_end, y_pos), color, 1, cv2.LINE_AA)
            # 绘制间距下边界
            cv2.line(img, (x, y_pos + gap_size), (x_end, y_pos + gap_size), color, 1, cv2.LINE_AA)
            x = x_end + gap_length
        
        # 如果提供了切割位置，画一条实线标记实际切割位置
        if cut_positions and i < len(cut_positions):
            cv2.line(img, (0, cut_positions[i]), (img.shape[1], cut_positions[i]), color, 2, cv2.LINE_AA)
        
        # 添加间距大小标注
        cv2.putText(img, f'Gap: {gap_size}px', (10, y_pos + gap_size//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return img

def get_gap_level(gap_size):
    """获取间距等级
    
    Args:
        gap_size: 间距大小
    
    Returns:
        level: 间距等级（1表示70-99像素，2表示100像素及以上）
    """
    return 2 if gap_size >= 100 else 1

def process_directory(input_dir=DIRECTORIES['OUTPUT_REGIONS']):
    """处理目录中的所有main和full图像
    """
    if not os.path.exists(input_dir):
        print(f"错误：输入目录 {input_dir} 不存在！")
        return
    
    # 创建输出目录
    output_dir = 'output_gaps'
    sections_dir = os.path.join(output_dir, 'sections')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sections_dir, exist_ok=True)
    
    # 查找所有main和full图像
    image_files = []
    for pattern in ['*main*.png', '*full*.png']:
        image_files.extend(glob.glob(os.path.join(input_dir, pattern)))
    
    if not image_files:
        print(f"在 {input_dir} 中没有找到main或full图像！")
        return
    
    print("==> 开始检测段落间距...")
    
    for image_path in image_files:
        base_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(base_name)[0]
        print(f"正在处理: {base_name}")
        
        # 检测间距
        gaps = detect_large_gaps(image_path, min_gap=70, max_gap=100)
        
        if gaps:
            # 读取原始图像
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"无法读取图像: {image_path}")
                continue
                
            # 切割图像并获取切割位置
            sections, cut_positions = split_by_gaps(img, gaps)
            
            # 标记间距和切割位置
            marked_img = mark_gaps(image_path, gaps, cut_positions)
            if marked_img is not None:
                # 保存标记后的图像
                output_path = os.path.join(output_dir, f'gaps_{base_name}')
                cv2.imencode('.png', marked_img)[1].tofile(output_path)
                
                # 保存切割后的段落
                for i, section in enumerate(sections):
                    # 获取当前段落对应的间距等级
                    gap_level = None
                    if i < len(gaps):
                        _, gap_size, _ = gaps[i]
                        gap_level = get_gap_level(gap_size)
                    
                    # 构建输出文件名
                    if gap_level is not None:
                        section_name = f'{name_without_ext}_section_{i+1}_gap{gap_level}.png'
                    else:
                        # 最后一个段落使用前一个间距的等级
                        if gaps:
                            _, last_gap_size, _ = gaps[-1]
                            last_gap_level = get_gap_level(last_gap_size)
                            section_name = f'{name_without_ext}_section_{i+1}_gap{last_gap_level}.png'
                        else:
                            section_name = f'{name_without_ext}_section_{i+1}.png'
                    
                    section_path = os.path.join(sections_dir, section_name)
                    cv2.imencode('.png', section)[1].tofile(section_path)
                
                print(f"找到 {len(gaps)} 个大间距")
                print(f"原图高度: {img.shape[0]}")
                print(f"切割后段落总高度: {sum(section.shape[0] for section in sections)}")
                for i, (y_pos, gap_size, is_extra_large) in enumerate(gaps):
                    color_text = "蓝色" if is_extra_large else "红色"
                    cut_pos = cut_positions[i] if i < len(cut_positions) else None
                    gap_level = get_gap_level(gap_size)
                    print(f"  位置: {y_pos}, 间距: {gap_size}px ({color_text}, 等级{gap_level}), 切割位置: {cut_pos}")
                print(f"已生成 {len(sections)} 个图像段落")
        else:
            print("未检测到大间距")
            # 保存完整图像作为一个段落
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                section_path = os.path.join(sections_dir, 
                                          f'{name_without_ext}_section_1.png')
                cv2.imencode('.png', img)[1].tofile(section_path)
                print("已保存完整图像作为单个段落")
    
    print("==> 处理完成！")

if __name__ == "__main__":
    process_directory() 