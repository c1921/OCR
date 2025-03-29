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
    'OUTPUT_REGIONS': 'output_regions',  # 区域分割的输出目录
    'OUTPUT_SECTIONS': 'output_sections',  # 段落分割的输出目录
    'OUTPUT_MARKED': 'output_marked',  # 标记后的图像
}

# 图像格式
IMAGE_EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.bmp']

# 标记颜色 (BGR格式)
COLORS = {
    'SPACE_MARK': (0, 255, 0),  # 行间距标记颜色（绿色）
    'INDENT_MARK': (0, 0, 255),  # 缩进标记颜色（红色）
    'SEPARATOR_MARK': (255, 0, 0),  # 分隔线标记颜色（蓝色）
    'CORNER_MARK': (255, 165, 0),  # 角标标记颜色（橙色）
    'PAGE_CUT_MARK': (255, 0, 255),  # 页码裁切线和区域标记颜色（紫色）
}

# 标记线配置
MARK_LINE = {
    'THICKNESS': 2,  # 线宽
    'VERTICAL_OFFSET': 2  # 垂直偏移
}

# 添加分隔线检测相关配置
SEPARATOR = {
    'MIN_LENGTH': 400,  # 分隔线最小长度
    'MAX_LENGTH': 600,  # 分隔线最大长度
    'MAX_HEIGHT': 3,    # 分隔线最大高度
    'MARGIN': 10,       # 切分时的边距
    'MIN_SECTION_HEIGHT': 30,  # 最小文本段落高度
}

# 角标检测相关配置
CORNER = {
    'WIDTH': 150,       # 角落区域宽度
    'HEIGHT': 150,      # 角落区域高度
    'MARGIN': 10,       # 切除时的边距
    'TOP_LINE': 298,    # 顶部角标检测线
    'TOP_RADIUS': 60,   # 顶部角标检测范围
    'MAX_HEIGHT': 50    # 顶部角标最大高度
}

# 页码检测相关配置
PAGE_NUMBER = {
    'RADIUS': 60,      # 检测区域半径
    'MAX_WIDTH': 120,    # 页码最大宽度
    'MAX_HEIGHT': 50,   # 页码最大高度
    'MARGIN': 5,        # 切除时的边距
} 