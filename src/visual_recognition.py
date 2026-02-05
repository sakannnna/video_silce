import os
import logging
import json
import hashlib
from http import HTTPStatus
import dashscope
from dashscope import MultiModalConversation

# 导入配置
try:
    from config import DASHSCOPE_API_KEY, QWEN_VL_MODEL
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DASHSCOPE_API_KEY, QWEN_VL_MODEL

logger = logging.getLogger(__name__)

class VisualRecognition:
    def __init__(self):
        self.api_key = DASHSCOPE_API_KEY
        if not self.api_key:
            logger.warning("未找到 DASHSCOPE_API_KEY")
        dashscope.api_key = self.api_key
        self.model = QWEN_VL_MODEL
        
        # 初始化缓存
        self.cache_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "api_cache.json")
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}")
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    def _get_file_hash(self, file_path):
        """计算文件内容的MD5哈希"""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                # 读取前64KB和后64KB (大文件优化)，或者全读 (图片通常不大)
                # 图片比较小，直接全读
                buf = f.read()
                hasher.update(buf)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"计算哈希失败: {e}")
            return None

    def analyze_image(self, image_path, prompt="请详细描述这张图片的内容，包括场景、人物、动作和关键视觉元素。"):
        """
        分析单张图片 (带缓存)
        Args:
            image_path: 图片路径
            prompt: 提示词
        Returns:
            str: 图片描述
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return None

        # 1. 检查缓存
        file_hash = self._get_file_hash(image_path)
        if file_hash:
            cache_key = f"{file_hash}_{prompt}"
            if cache_key in self.cache:
                logger.info(f"Cache hit for {image_path}")
                return self.cache[cache_key]

        # 2. 准备API调用
        # Windows path to URI format requires correct handling
        abs_path = os.path.abspath(image_path)
        # Handle windows path separators
        if os.name == 'nt':
            abs_path = abs_path.replace('\\', '/')
        img_uri = f"file://{abs_path}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": img_uri},
                    {"text": prompt}
                ]
            }
        ]

        try:
            logger.info(f"Calling VL model for {image_path}")
            response = MultiModalConversation.call(model=self.model, messages=messages)
            
            result_text = None
            
            if response.status_code == HTTPStatus.OK:
                # 解析响应
                content = response.output.choices[0].message.content
                
                if isinstance(content, list):
                    text_content = ""
                    for item in content:
                        if 'text' in item:
                            text_content += item['text']
                    result_text = text_content
                elif isinstance(content, str):
                    result_text = content
                else:
                    logger.warning(f"Unexpected content format: {type(content)}")
                    result_text = str(content)
                
                # 3. 写入缓存
                if result_text and file_hash:
                    cache_key = f"{file_hash}_{prompt}"
                    self.cache[cache_key] = result_text
                    self._save_cache()
                
                return result_text
            else:
                logger.error(f"API Error: {response.code} - {response.message}")
                return None
        except Exception as e:
            logger.error(f"Exception during visual recognition: {e}")
            return None
