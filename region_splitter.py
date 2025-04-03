import glob
import cv2
import numpy as np
import os
import yaml
import logging
from datetime import datetime

def setup_logging():
    """设置日志配置"""
    # 创建logs目录
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 设置日志文件名（使用时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/region_splitter_{timestamp}.log'
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    return logging.getLogger(__name__)

# 创建logger
logger = setup_logging()

def load_config():
    """加载配置文件并转换配置项为大写格式"""
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 转换需要大写的配置项
        uppercase_keys = ['directories', 'colors', 'mark_line', 
                         'separator', 'corner', 'page_number']
        
        for key in uppercase_keys:
            if key in config:
                config[key] = {k.upper(): v for k, v in config[key].items()}
                
        return config
    except Exception as e:
        print(f"加载配置文件失败: {str(e)}")
        return None

# 加载配置
config = load_config()
if not config:
    raise ValueError("无法加载配置文件")

# 从配置中获取常量
DIRECTORIES = config['directories']
IMAGE_EXTENSIONS = config['image_extensions']
COLORS = config['colors']
MARK_LINE = config['mark_line']
SEPARATOR = config['separator']
CORNER = config['corner']
PAGE_NUMBER = config['page_number']

def detect_separator_lines(image_path):
    """检测注释分隔线"""
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)  # 直接读取为灰度图
    if img is None:
        print(f"无法读取图像: {image_path}")
        return []
    
    # 使用自适应阈值进行二值化
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # 使用形态学操作突出横线
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                                (SEPARATOR['MIN_LENGTH'], 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    
    # 查找轮廓并筛选
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_SIMPLE)
    
    return sorted(y for x, y, w, h in [cv2.boundingRect(cnt) for cnt in contours]
                 if SEPARATOR['MIN_LENGTH'] <= w <= SEPARATOR['MAX_LENGTH'] 
                 and h <= SEPARATOR['MAX_HEIGHT'])

def process_roi(binary, roi_coords, offset=(0, 0), size_limits=None):
    """处理感兴趣区域(ROI)中的文本
    
    Args:
        binary: 二值化图像
        roi_coords: ROI坐标 (x1, y1, x2, y2)
        offset: 坐标偏移量
        size_limits: 文本块大小限制 (max_width, max_height)
    
    Returns:
        corners: 检测到的文本区域列表
    """
    x1, y1, x2, y2 = roi_coords
    roi = binary[y1:y2, x1:x2]
    
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corners = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if size_limits and (w > size_limits[0] or h > size_limits[1]):
            continue
            
        # 转换到原图坐标并添加边距
        x_min = max(0, x + offset[0] - CORNER['MARGIN'])
        y_min = max(0, y + offset[1] - CORNER['MARGIN'])
        x_max = min(binary.shape[1], x + w + offset[0] + CORNER['MARGIN'])
        y_max = min(binary.shape[0], y + h + offset[1] + CORNER['MARGIN'])
        
        corners.append(((x_min, y_min), (x_max, y_max)))
    
    return corners

def detect_corner_text(img):
    """检测顶部角标和页码"""
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    corners = []
    page_number_found = False
    
    # 1. 检测顶部角标 - 在y=298附近
    y_line = CORNER['TOP_LINE']
    y_min = max(0, y_line - CORNER['TOP_RADIUS'])
    y_max = min(height, y_line + CORNER['TOP_RADIUS'])
    top_area = binary[y_min:y_max, :]
    
    # 查找这个区域中的所有文本
    contours, _ = cv2.findContours(top_area, cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 找到所有轮廓的边界
        x_min, y_min = width, top_area.shape[0]
        x_max, y_max = 0, 0
        
        # 合并所有可能的角标部分
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # 只处理合适高度的文本
            if h < CORNER['MAX_HEIGHT']:
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
        
        # 如果找到了合适的文本区域
        if x_max > x_min:
            # 转换到原图坐标并添加边距
            x_min = max(0, x_min - CORNER['MARGIN'])
            y_min = max(0, y_min + y_line - CORNER['TOP_RADIUS'] - CORNER['MARGIN'])
            x_max = min(width, x_max + CORNER['MARGIN'])
            y_max = min(height, y_max + y_line - CORNER['TOP_RADIUS'] + CORNER['MARGIN'])
            
            corners.append(('corner', ((x_min, y_min), (x_max, y_max))))
    
    # 2. 检测页码 - 作为一个整体
    y_line = 2748
    y_min = max(0, y_line - PAGE_NUMBER['RADIUS'])
    y_max = min(height, y_line + PAGE_NUMBER['RADIUS'])
    page_area = binary[y_min:y_max, :]
    
    # 查找这个区域中的所有文本
    contours, _ = cv2.findContours(page_area, cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 找到所有轮廓的边界
        x_min, y_min = width, page_area.shape[0]
        x_max, y_max = 0, 0
        
        # 合并所有可能的页码部分
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # 只处理合适高度的文本
            if h < PAGE_NUMBER['MAX_HEIGHT']:
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
        
        # 如果找到了合适的文本区域且总宽度合适
        if x_max > x_min and (x_max - x_min) <= PAGE_NUMBER['MAX_WIDTH']:
            page_number_found = True
            # 转换到原图坐标并添加边距
            x_min = max(0, x_min - PAGE_NUMBER['MARGIN'])
            y_min = max(0, y_min + y_line - PAGE_NUMBER['RADIUS'] - PAGE_NUMBER['MARGIN'])
            x_max = min(width, x_max + PAGE_NUMBER['MARGIN'])
            y_max = min(height, y_max + y_line - PAGE_NUMBER['RADIUS'] + PAGE_NUMBER['MARGIN'])
            
            corners.append(('page', ((x_min, y_min), (x_max, y_max))))
    
    return corners, page_number_found

def mark_detected_regions(img, corners, separator_positions):
    """标记检测到的区域和分隔线
    
    Args:
        img: 输入图像
        corners: 检测到的角标和页码区域
        separator_positions: 分隔线位置列表
    
    Returns:
        marked_img: 标记后的图像
        page_number_y: 页码位置的y坐标
        top_corner_y: 顶部角标的底部y坐标
    """
    marked_img = img.copy()
    page_number_y = float('inf')
    top_corner_y = float('inf')
    
    # 找到最上方的页码位置和顶部角标位置
    for type_, ((_, y_min), (_, y_max)) in corners:
        if type_ == 'page':
            page_number_y = min(page_number_y, y_min)
        elif type_ == 'corner':
            top_corner_y = min(top_corner_y, y_max)
    
    # 标记检测到的区域
    for type_, (top_left, bottom_right) in corners:
        if type_ == 'page' and top_left[1] == page_number_y:
            # 标记页码裁切线和页码区域
            cv2.line(marked_img, 
                    (0, top_left[1]), 
                    (marked_img.shape[1], top_left[1]),
                    COLORS['PAGE_CUT_MARK'], 
                    MARK_LINE['THICKNESS'])
            cv2.rectangle(marked_img, top_left, bottom_right,
                        COLORS['PAGE_CUT_MARK'],
                        MARK_LINE['THICKNESS'])
        elif type_ == 'corner' and bottom_right[1] == top_corner_y:
            # 标记顶部角标裁切线和区域
            cv2.line(marked_img, 
                    (0, bottom_right[1]), 
                    (marked_img.shape[1], bottom_right[1]),
                    COLORS['CORNER_MARK'], 
                    MARK_LINE['THICKNESS'])
            cv2.rectangle(marked_img, top_left, bottom_right,
                        COLORS['CORNER_MARK'],
                        MARK_LINE['THICKNESS'])
    
    # 标记分隔线
    for pos in separator_positions:
        if not page_number_y or pos < page_number_y:
            cv2.line(marked_img, (0, pos), (marked_img.shape[1], pos),
                    COLORS['SEPARATOR_MARK'], MARK_LINE['THICKNESS'])
    
    # 重置无限值
    if page_number_y == float('inf'):
        page_number_y = None
    if top_corner_y == float('inf'):
        top_corner_y = None
    
    return marked_img, page_number_y, top_corner_y

def process_image_sections(img, separator_positions, page_number_y, top_corner_y, output_dir, base_name):
    """处理图像分段
    
    Args:
        img: 输入图像
        separator_positions: 分隔线位置列表
        page_number_y: 页码位置的y坐标
        top_corner_y: 顶部角标的底部y坐标
        output_dir: 输出目录
        base_name: 输出文件基础名
    """
    def save_section(start_y, end_y, section_type, index):
        """保存图像段落"""
        if end_y - start_y <= SEPARATOR['MIN_SECTION_HEIGHT']:
            return
        
        section = img[start_y:end_y]
        output_path = os.path.join(output_dir, 
                                 f'{base_name}_{section_type}_{index}.png')
        cv2.imencode('.png', section)[1].tofile(output_path)
    
    # 处理图像段落
    if not separator_positions:
        # 确定结束位置（页码或图像底部）
        end_y = page_number_y if page_number_y else img.shape[0]
        # 从顶部角标下方开始
        start_y = top_corner_y if top_corner_y else 0
        save_section(start_y, end_y, 'full', 1)
    else:
        # 从顶部角标下方开始处理
        last_end = top_corner_y if top_corner_y else 0
        for i, pos in enumerate(separator_positions):
            # 如果分隔线在页码之后，跳过处理
            if page_number_y and pos >= page_number_y:
                break
            
            # 处理正文
            save_section(last_end, pos-SEPARATOR['MARGIN'], 'main', i+1)
            
            # 处理注释
            next_pos = min(filter(lambda x: x is not None, [
                separator_positions[i + 1] if i < len(separator_positions) - 1 else None,
                page_number_y,
                img.shape[0]
            ]))
            save_section(pos+SEPARATOR['MARGIN'], next_pos, 'note', i+1)
            
            last_end = pos

class ImageProcessor:
    """图像处理流水线
    
    处理步骤：
    1. 加载图像
    2. 检测区域（顶部角标和页码）
    3. 检测分隔线
    4. 标记检测结果
    5. 分割并保存图像
    """
    
    def __init__(self, image_path, output_dir=DIRECTORIES['OUTPUT_REGIONS']):
        self.image_path = image_path
        self.output_dir = output_dir
        self.base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 初始化成员变量
        self.img = None
        self.corners = None
        self.separator_positions = None
        self.marked_img = None
        self.page_number_y = None
        self.top_corner_y = None
    
    def load_image(self):
        """步骤1：加载图像"""
        self.img = cv2.imdecode(np.fromfile(self.image_path, dtype=np.uint8), 
                              cv2.IMREAD_COLOR)
        if self.img is None:
            raise ValueError(f"无法读取图像: {self.image_path}")
        return self
    
    def detect_regions(self):
        """步骤2：检测顶部角标和页码区域"""
        self.corners, self.page_number_found = detect_corner_text(self.img)
        return self
    
    def detect_separators(self):
        """步骤3：检测分隔线"""
        self.separator_positions = detect_separator_lines(self.image_path)
        return self
    
    def mark_regions(self):
        """步骤4：标记检测到的区域"""
        self.marked_img, self.page_number_y, self.top_corner_y = mark_detected_regions(
            self.img, self.corners, self.separator_positions)
        return self
    
    def save_sections(self):
        """步骤5：分割并保存图像"""
        # 创建输出目录和标记目录
        os.makedirs(self.output_dir, exist_ok=True)
        marked_dir = os.path.join(self.output_dir, 'marked')
        os.makedirs(marked_dir, exist_ok=True)
        
        # 处理图像分段
        process_image_sections(
            self.img, 
            self.separator_positions, 
            self.page_number_y, 
            self.top_corner_y,
            self.output_dir, 
            self.base_name
        )
        
        # 保存标记图像到marked子目录
        cv2.imencode('.png', self.marked_img)[1].tofile(
            os.path.join(marked_dir, f'{self.base_name}_marked.png'))
        return self
    
    def process(self):
        """执行完整的处理流水线"""
        try:
            result = (self.load_image()
                         .detect_regions()
                         .detect_separators()
                         .mark_regions()
                         .save_sections())
            
            # 检查是否检测到页码
            if not any(type_ == 'page' for type_, _ in self.corners):
                logger.warning(f"未检测到页码: {self.base_name}")
            
            return result
            
        except Exception as e:
            logger.error(f"处理图像 {self.image_path} 时出错: {str(e)}")
            return None

def split_text_and_notes(image_path, output_dir=DIRECTORIES['OUTPUT_REGIONS']):
    """分离正文和注释，去除角标和页码"""
    processor = ImageProcessor(image_path, output_dir)
    return processor.process()

def process_directory(input_dir=DIRECTORIES['INPUT_DIR']):
    """处理目录中的所有图像"""
    if not os.path.exists(input_dir):
        logger.error(f"输入目录 {input_dir} 不存在！")
        return
    
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_files:
        logger.error(f"在 {input_dir} 目录下没有找到图像文件！")
        return
    
    # 自然排序
    def natural_sort_key(s):
        """用于自然排序的键函数"""
        import re
        # 将字符串中的数字转换为整数，用于正确排序
        return [int(text) if text.isdigit() else text.lower()
               for text in re.split('([0-9]+)', s)]
    
    # 对文件进行自然排序
    image_files.sort(key=natural_sort_key)
    
    logger.info("==> 开始分离正文与注释...")
    
    for image_path in image_files:
        logger.info(f"正在处理: {os.path.basename(image_path)}")
        split_text_and_notes(image_path)
    
    logger.info("==> 处理完成！")

if __name__ == "__main__":
    process_directory() 