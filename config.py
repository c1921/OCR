# 图像处理相关参数
IMAGE_PROCESSING = {
    'MIN_LINE_HEIGHT': 10,  # 最小行高
    'GAUSSIAN_BLUR_KERNEL': (5, 5),  # 高斯模糊核大小
    'PROJECTION_THRESHOLD': 0.05,  # 投影阈值比例
    'LINE_MARGIN': 5,  # 行边距
    'WINDOW_SIZE': 10,  # 滑动窗口大小
    'TEXT_THRESHOLD': 0.15,  # 文本检测阈值
}

# 段落检测相关参数
PARAGRAPH_DETECTION = {
    'SPACE_MEDIAN_RATIO': 1.3,  # 行间距中位数比例
    'SPACE_MEAN_STD_RATIO': 0.5,  # 行间距均值标准差比例
}

# 文件和目录配置
DIRECTORIES = {
    'INPUT_DIR': 'img_input',
    'OUTPUT_PARAGRAPHS': 'output_paragraphs',
    'OUTPUT_MARKED': 'output_marked'
}

# 图像格式
IMAGE_EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.bmp']

# 标记颜色 (BGR格式)
COLORS = {
    'SPACE_MARK': (0, 255, 0),  # 行间距标记颜色（绿色）
    'INDENT_MARK': (0, 0, 255)  # 缩进标记颜色（红色）
}

# 标记线配置
MARK_LINE = {
    'THICKNESS': 2,  # 线宽
    'VERTICAL_OFFSET': 2  # 垂直偏移
} 