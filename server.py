# server.py
"""
server.py - 视频智能剪辑工具后端服务
基于最新 main.py 改造，通过函数参数传递替代交互式输入
"""

import os
import sys
import json
import logging
import glob
import asyncio
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入自定义模块
from config import (
    INPUT_VIDEO_DIR, OUTPUT_VIDEO_DIR, PROCESSED_AUDIO_DIR,
    TRANSCRIPTS_DIR, ANALYSIS_RESULTS_DIR, SLICE_VIDEO_DIR, 
    KEYFRAMES_DIR, RAGSCRIPTS_DIR, VERTICAL_VIDEO_DIR, VIDEO_POOL_DIR
)
from src.video_processor import VideoProcessor
from src.speech_to_text import SpeechToText
from src.text_analyzer import TextAnalyzer
from src.visual_recognition import VisualRecognition
from src.data_merger import merge_audio_visual_data
from src.data_cleaner import clean_json_data
from src.rag_engine import VideoKnowledgeBase
from src.utils import get_file_hash, ensure_in_video_pool
import numpy as np
from PIL import Image

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 辅助函数 ====================

def ensure_directories():
    """确保所有必要的目录都存在"""
    directories = [
        INPUT_VIDEO_DIR,
        OUTPUT_VIDEO_DIR,
        PROCESSED_AUDIO_DIR,
        TRANSCRIPTS_DIR,
        ANALYSIS_RESULTS_DIR,
        SLICE_VIDEO_DIR,
        KEYFRAMES_DIR,
        VERTICAL_VIDEO_DIR,
        VIDEO_POOL_DIR
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"确保目录存在: {directory}")
    
    # 创建RAG目录
    Path(RAGSCRIPTS_DIR).mkdir(parents=True, exist_ok=True)
    
    return True

def save_transcript(transcript, video_name):
    """保存转录文本到文件"""
    try:
        transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{video_name}_transcript.json")
        logger.info(f"准备保存转录结果到 {transcript_path}")
        
        # 确保转录是JSON可序列化的
        if isinstance(transcript, list):
            # 如果是单词列表，转换为标准格式
            serializable_transcript = []
            for item in transcript:
                if isinstance(item, dict):
                    serializable_transcript.append(item)
                else:
                    # 尝试转换为字典
                    serializable_transcript.append({"word": str(item)})
        else:
            serializable_transcript = str(transcript)
        
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_transcript, f, ensure_ascii=False, indent=2)
        
        logger.info(f"转录文本已保存到: {transcript_path}")
        return transcript_path
    except Exception as e:
        logger.error(f"保存转录结果失败: {str(e)}")
        return None

def save_analysis_results(segments, video_name, user_instruction):
    """保存分析结果到文件"""
    try:
        results = {
            "video_name": video_name,
            "user_instruction": user_instruction,
            "segments": segments,
            "total_segments": len(segments),
            "total_duration": sum(seg["end_time"] - seg["start_time"] for seg in segments if "start_time" in seg and "end_time" in seg)
        }
        
        results_path = os.path.join(ANALYSIS_RESULTS_DIR, f"{video_name}_analysis.json")
        logger.info(f"准备保存分析结果到 {results_path}")
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分析结果已保存到: {results_path}")
        return results_path
    except Exception as e:
        logger.error(f"保存分析结果失败: {str(e)}")
        return None

def calculate_image_difference(img1_path, img2_path):
    """计算两张图片的差异 (MSE)"""
    try:
        # Resize to small size for fast comparison
        with Image.open(img1_path) as i1, Image.open(img2_path) as i2:
            i1 = i1.resize((64, 64)).convert('L')
            i2 = i2.resize((64, 64)).convert('L')
            arr1 = np.array(i1)
            arr2 = np.array(i2)
            mse = np.mean((arr1 - arr2) ** 2)
            return mse
    except Exception as e:
        logger.warning(f"图片差异计算失败: {e}")
        return float('inf')

# ==================== 文件列表获取函数 ====================

def get_video_files() -> List[str]:
    """获取输入目录中的所有视频文件"""
    video_files = []
    if os.path.exists(INPUT_VIDEO_DIR):
        for file in os.listdir(INPUT_VIDEO_DIR):
            if file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                video_files.append(file)
    return sorted(video_files)

def get_transcript_files() -> List[str]:
    """获取转录目录中的所有JSON文件"""
    json_files = []
    if os.path.exists(TRANSCRIPTS_DIR):
        for file in os.listdir(TRANSCRIPTS_DIR):
            if file.lower().endswith('.json'):
                json_files.append(file)
    return sorted(json_files)

def get_rag_files() -> List[str]:
    """获取RAG目录中的RAG文件"""
    rag_files = []
    if os.path.exists(RAGSCRIPTS_DIR):
        for file in os.listdir(RAGSCRIPTS_DIR):
            if file.lower().endswith('_rag.json'):
                rag_files.append(file)
    return sorted(rag_files)

def get_analysis_files() -> List[str]:
    """获取分析结果目录中的文件"""
    analysis_files = []
    if os.path.exists(ANALYSIS_RESULTS_DIR):
        for file in os.listdir(ANALYSIS_RESULTS_DIR):
            if file.lower().endswith('_analysis.json'):
                analysis_files.append(file)
    return sorted(analysis_files)

def get_video_md5(video_filename: str) -> Optional[str]:
    """获取视频文件的MD5"""
    video_path = os.path.join(INPUT_VIDEO_DIR, video_filename)
    if os.path.exists(video_path):
        return get_file_hash(video_path)
    return None

# ==================== 核心功能函数 ====================

def data_processing(video_filename: str) -> Dict[str, Any]:
    """
    数据准备功能：提取音频、语音转文字、视觉分析
    
    Args:
        video_filename: 视频文件名
        
    Returns:
        包含处理结果的字典
    """
    result = {
        "success": False,
        "message": "",
        "video_name": "",
        "video_md5": "",
        "transcript_path": None,
        "transcript_count": 0,
        "keyframes_count": 0,
        "visual_segments_count": 0
    }
    
    try:
        logger.info("开始进行数据处理")
        logger.info(f"处理视频: {video_filename}")
        
        video_path = os.path.join(INPUT_VIDEO_DIR, video_filename)
        
        if not os.path.exists(video_path):
            error_msg = f"视频文件不存在: {video_path}"
            logger.error(error_msg)
            result["message"] = error_msg
            return result
        
        # 计算 MD5 并入池
        video_md5, pool_path = ensure_in_video_pool(video_path, VIDEO_POOL_DIR)
        logger.info(f"视频 MD5: {video_md5}")
        
        video_name = os.path.splitext(video_filename)[0]
        result["video_name"] = video_name
        result["video_md5"] = video_md5
        
        # 2. 初始化处理器
        logger.info("初始化处理器")
        video_processor = VideoProcessor()
        speech_to_text = SpeechToText()
        visual_recognition = VisualRecognition()
        
        # 3. 提取音频
        logger.info("提取音频")
        audio_filename = f"{video_name}.wav"
        audio_path = os.path.join(PROCESSED_AUDIO_DIR, audio_filename)
        
        success = video_processor.extract_audio(video_path, audio_path)
        if not success:
            error_msg = "音频提取失败"
            logger.error(error_msg)
            result["message"] = error_msg
            return result
        
        # 4. 语音转文字
        logger.info("语音转文字")
        first_transcript = speech_to_text.transcribe(audio_path, video_md5=video_md5)
        if not first_transcript:
            error_msg = "语音转文字失败"
            logger.error(error_msg)
            result["message"] = error_msg
            return result
        
        transcript = speech_to_text.split_by_punctuation(first_transcript)
        result["transcript_count"] = len(transcript)
        
        # 5. 视觉内容分析
        logger.info("视觉内容分析")
        
        # 提取关键帧
        kf_output_dir = os.path.join(KEYFRAMES_DIR, video_name)
        Path(kf_output_dir).mkdir(parents=True, exist_ok=True)
        
        keyframes = video_processor.extract_keyframes(video_path, kf_output_dir, interval=2.0)
        logger.info(f"提取了 {len(keyframes)} 个潜在关键帧")
        
        # 关键帧去重
        visual_segments = []
        last_processed_kf_path = None
        MSE_THRESHOLD = 50.0
        
        unique_keyframes = []
        skipped_count = 0
        
        logger.info("正在进行关键帧去重...")
        for kf in keyframes:
            kf_path = kf['path']
            
            if last_processed_kf_path:
                mse = calculate_image_difference(last_processed_kf_path, kf_path)
                if mse < MSE_THRESHOLD:
                    skipped_count += 1
                    continue
            
            unique_keyframes.append(kf)
            last_processed_kf_path = kf_path
        
        result["keyframes_count"] = len(unique_keyframes)
        logger.info(f"去重完成: 共有 {len(unique_keyframes)} 帧待分析, 跳过 {skipped_count} 帧")
        
        # 异步批量分析
        if unique_keyframes:
            logger.info("开始异步调用视觉模型分析关键帧...")
            
            try:
                async def process_images_async_with_progress(visual_recognition, unique_keyframes):
                    from tqdm import tqdm
                    
                    total_keyframes = len(unique_keyframes)
                    descriptions = [None] * total_keyframes
                    sem = asyncio.Semaphore(15)
                    pbar = tqdm(total=total_keyframes, desc="视觉分析进度", unit="帧")
                    
                    async def bounded_analyze_wrapper(index, kf):
                        async with sem:
                            try:
                                res = await visual_recognition.analyze_image_async(kf['path'], auto_save=False)
                            except Exception as e:
                                logger.error(f"Error analyzing frame {index}: {e}")
                                res = None
                            finally:
                                pbar.update(1)
                            return index, res

                    tasks = [bounded_analyze_wrapper(i, kf) for i, kf in enumerate(unique_keyframes)]
                    
                    for task in asyncio.as_completed(tasks):
                        idx, res = await task
                        descriptions[idx] = res
                        
                        # 每完成 50 个保存一次缓存
                        if (pbar.n) % 50 == 0 and hasattr(visual_recognition, 'save_cache'):
                             await visual_recognition.save_cache()
                    
                    pbar.close()
                    
                    if hasattr(visual_recognition, 'save_cache'):
                        await visual_recognition.save_cache()
                        
                    return descriptions

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                descriptions = loop.run_until_complete(
                    process_images_async_with_progress(visual_recognition, unique_keyframes)
                )
                loop.close()
                
            except Exception as e:
                logger.error(f"异步分析出错: {e}")
                descriptions = [None] * len(unique_keyframes)

            analyzed_count = 0
            for kf, description in zip(unique_keyframes, descriptions):
                timestamp = kf['time']
                if description:
                    visual_segments.append({
                        "word": f"[视觉画面: {description}]", 
                        "text": f"[视觉画面: {description}]",
                        "start": timestamp,
                        "end": timestamp + 2.0
                    })
                    analyzed_count += 1
                else:
                    logger.warning(f"Failed to analyze frame at {timestamp}")
            
            result["visual_segments_count"] = analyzed_count
            logger.info(f"视觉分析完成: 成功分析 {analyzed_count} 帧")
        
        # 整合结果
        full_transcript = merge_audio_visual_data(transcript, visual_segments)
        logger.info(f"结果整合完成，共 {len(full_transcript)} 条记录")
        
        # 保存转录结果
        transcript_path = save_transcript(full_transcript, video_name)
        result["transcript_path"] = transcript_path
        
        result["success"] = True
        result["message"] = f"数据处理完成，共生成 {len(full_transcript)} 条记录"
        
    except Exception as e:
        logger.exception(f"数据处理出错: {str(e)}")
        result["message"] = f"数据处理出错: {str(e)}"
    
    return result

def rag_building(rag_filename: Optional[str] = None, 
                 source_json: Optional[str] = None,
                 category: str = "general",
                 lib_name: str = "default_lib") -> Dict[str, Any]:
    """
    RAG构建功能：清洗数据并构建知识库
    
    Args:
        rag_filename: 可选的RAG文件名，如果提供则直接使用
        source_json: 可选的源JSON文件名，如果提供则先清洗
        category: 分类标签
        lib_name: 逻辑库名称
        
    Returns:
        包含处理结果的字典
    """
    result = {
        "success": False,
        "message": "",
        "rag_filename": None,
        "total_items": 0,
        "collection_count": 0,
        "lib_name": lib_name,
        "video_md5": None
    }
    
    try:
        logger.info("开始RAG数据准备与测试")
        
        # 如果提供了源JSON，先清洗
        if source_json:
            json_path = os.path.join(TRANSCRIPTS_DIR, source_json)
            if not os.path.exists(json_path):
                result["message"] = f"JSON文件不存在: {json_path}"
                return result
            
            rag_filename = source_json.replace(".json", "_rag.json")
            rag_path = os.path.join(RAGSCRIPTS_DIR, rag_filename)
            
            logger.info(f"清洗数据: {source_json} -> {rag_filename}")
            clean_json_data(json_path, rag_path, category_tag=category)
            result["rag_filename"] = rag_filename
        
        # 如果没有指定RAG文件名，使用最新的
        if not rag_filename:
            rag_files = get_rag_files()
            if not rag_files:
                result["message"] = "没有找到RAG文件"
                return result
            rag_filename = rag_files[-1]  # 使用最新的
        
        rag_path = os.path.join(RAGSCRIPTS_DIR, rag_filename)
        if not os.path.exists(rag_path):
            result["message"] = f"RAG文件不存在: {rag_path}"
            return result
        
        # 加载清洗后的数据
        with open(rag_path, 'r', encoding='utf-8') as f:
            rag_data = json.load(f)
        
        logger.info(f"RAG数据加载完成，共 {len(rag_data)} 条")
        result["total_items"] = len(rag_data)
        
        # 获取 Video MD5
        base_name = rag_filename.replace("_rag.json", "").replace(".json", "")
        video_md5 = None
        
        # 尝试从 INPUT_VIDEO_DIR 找对应视频
        candidates = glob.glob(os.path.join(INPUT_VIDEO_DIR, f"{base_name}.*"))
        for c in candidates:
            if c.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                video_md5 = get_file_hash(c)
                logger.info(f"找到对应视频: {c}, MD5: {video_md5}")
                break
        
        if not video_md5:
            logger.warning("未找到原始视频文件，使用文件名生成的伪MD5")
            video_md5 = hashlib.md5(base_name.encode()).hexdigest()
        
        result["video_md5"] = video_md5
        
        try:
            vkb = VideoKnowledgeBase(lib_name=lib_name)
            existing_count = vkb.collection.count()
            result["collection_count"] = existing_count
            logger.info(f"逻辑库 '{lib_name}' 中已有 {existing_count} 条数据")
            
            # 分批处理以避免 Embedding API 限制
            BATCH_SIZE = 20
            total_items = len(rag_data)
            logger.info(f"准备入库 {total_items} 条数据，分批处理中...")
            
            for i in range(0, total_items, BATCH_SIZE):
                batch_data = rag_data[i : i + BATCH_SIZE]
                vkb.add_data(batch_data, video_md5)
                logger.info(f"进度: {min(i + BATCH_SIZE, total_items)}/{total_items} 已处理")
            
            new_count = vkb.collection.count()
            result["collection_count"] = new_count
            result["success"] = True
            result["message"] = f"RAG知识库构建完成，逻辑库 '{lib_name}' 中共 {new_count} 条数据"
            
        except Exception as e:
            logger.error(f"RAG Error: {e}")
            result["message"] = f"RAG构建失败: {e}"
            return result
        
    except Exception as e:
        logger.exception(f"RAG处理出错: {e}")
        result["message"] = f"RAG处理出错: {e}"
    
    return result

def rag_search(query: str, 
               top_k: int = 3, 
               lib_name: str = "default_lib",
               expand_context: bool = True) -> Dict[str, Any]:
    """
    RAG搜索功能
    
    Args:
        query: 搜索查询
        top_k: 返回结果数量
        lib_name: 逻辑库名称
        expand_context: 是否扩展上下文
        
    Returns:
        包含搜索结果的字典
    """
    result = {
        "success": False,
        "message": "",
        "query": query,
        "results": []
    }
    
    try:
        vkb = VideoKnowledgeBase(lib_name=lib_name)
        
        logger.info(f"正在检索: '{query}'...")
        search_results = vkb.search(query, top_k=top_k, expand_context=expand_context)
        
        if search_results and 'documents' in search_results and search_results['documents']:
            for i, doc in enumerate(search_results['documents'][0]):
                if i < len(search_results['metadatas'][0]):
                    meta = search_results['metadatas'][0][i]
                    is_expanded = meta.get('is_expanded', False)
                    
                    result["results"].append({
                        "content": doc,
                        "start": meta.get('start', 0),
                        "end": meta.get('end', 0),
                        "type": meta.get('type', 'unknown'),
                        "category": meta.get('category', 'general'),
                        "video_md5": meta.get('video_md5', ''),
                        "is_expanded": is_expanded,
                        "raw_content": meta.get('raw_content', '')
                    })
            
            result["success"] = True
            result["message"] = f"找到 {len(result['results'])} 个结果"
        else:
            result["message"] = "未找到相关结果"
        
    except Exception as e:
        logger.error(f"搜索出错: {e}")
        result["message"] = f"搜索出错: {e}"
    
    return result

def video_editing(video_filename: str, 
                  user_instruction: str, 
                  max_duration: int) -> Dict[str, Any]:
    """
    视频剪辑功能
    
    Args:
        video_filename: 视频文件名
        user_instruction: 用户剪辑要求
        max_duration: 最大时长（秒）
        
    Returns:
        包含剪辑结果的字典
    """
    result = {
        "success": False,
        "message": "",
        "video_name": "",
        "segments": [],
        "selected_segments": [],
        "clip_paths": [],
        "output_path": None,
        "total_duration": 0
    }
    
    try:
        logger.info("开始执行视频剪辑及文本分析功能")
        
        video_path = os.path.join(INPUT_VIDEO_DIR, video_filename)
        if not os.path.exists(video_path):
            result["message"] = f"视频文件不存在: {video_path}"
            return result
        
        video_name = os.path.splitext(video_filename)[0]
        result["video_name"] = video_name
        
        # 初始化
        video_processor = VideoProcessor()
        text_analyzer = TextAnalyzer()
        
        # 读取转录文件
        transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{video_name}_transcript.json")
        if not os.path.exists(transcript_path):
            result["message"] = f"转录文件不存在: {transcript_path}，请先进行数据准备"
            return result
        
        logger.info("读取转录数据")
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
        
        # 分析文本
        logger.info("分析文本内容")
        segments = text_analyzer.analyze_transcript(transcript, user_instruction)
        result["segments"] = segments
        
        if not segments:
            logger.warning("未找到匹配的剪辑片段，使用默认剪辑")
            # 使用默认剪辑：前30秒
            segments = [{
                "start_time": 0.0,
                "end_time": min(30.0, max_duration),
                "reason": "默认剪辑：视频开头部分",
                "score": 5
            }]
        
        logger.info(f"文本分析完成，找到 {len(segments)} 个剪辑片段")
        
        # 选择关键片段
        logger.info("选择关键片段")
        selected_segments = video_processor.select_key_clips(segments, max_duration)
        result["selected_segments"] = selected_segments
        
        if not selected_segments:
            result["message"] = "未选择到有效的关键片段"
            return result
        
        logger.info(f"已选择 {len(selected_segments)} 个关键片段，总时长约 {max_duration} 秒")
        
        # 为每个片段添加序号
        for i, segment in enumerate(selected_segments):
            segment["clip_index"] = i + 1
        
        # 保存分析结果
        save_analysis_results(selected_segments, video_name, user_instruction)
        
        # 剪辑视频片段
        logger.info("剪辑视频片段")
        clip_paths = []
        
        for segment in selected_segments:
            if "start_time" not in segment or "end_time" not in segment:
                continue
            
            start_time = segment["start_time"]
            end_time = segment["end_time"]
            
            if end_time <= start_time:
                continue
            
            clip_filename = f"{video_name}_clip_{segment['clip_index']}.mp4"
            clip_path = os.path.join(SLICE_VIDEO_DIR, clip_filename)
            
            success = video_processor.create_clip(video_path, start_time, end_time, clip_path)
            if success:
                clip_paths.append(clip_path)
                logger.info(f"片段 {segment['clip_index']}: {start_time:.1f}s - {end_time:.1f}s")
        
        result["clip_paths"] = clip_paths
        
        if not clip_paths:
            result["message"] = "所有视频片段剪辑都失败"
            return result
        
        logger.info(f"共成功剪辑 {len(clip_paths)} 个片段")
        
        # 合并剪辑片段
        logger.info("合并剪辑片段")
        output_filename = f"{video_name}_edited.mp4"
        output_path = os.path.join(OUTPUT_VIDEO_DIR, output_filename)
        
        success = video_processor.combine_clips(clip_paths, output_path)
        if not success:
            result["message"] = "合并视频片段失败"
            return result
        
        # 计算总时长
        total_duration = 0
        for segment in selected_segments:
            if "start_time" in segment and "end_time" in segment:
                total_duration += segment["end_time"] - segment["start_time"]
        
        result["output_path"] = output_path
        result["total_duration"] = total_duration
        result["success"] = True
        result["message"] = f"视频剪辑完成，输出: {output_filename}"
        
        logger.info(f"视频处理完成，输出文件: {output_path}")
        
    except Exception as e:
        logger.exception(f"视频剪辑出错: {str(e)}")
        result["message"] = f"视频剪辑出错: {str(e)}"
    
    return result

def convert_to_vertical(video_filename: str, method: str = "solid") -> Dict[str, Any]:
    """
    横屏转竖屏功能
    
    Args:
        video_filename: 视频文件名
        method: 转换方法 ('solid', 'blur', 'static')
        
    Returns:
        包含转换结果的字典
    """
    result = {
        "success": False,
        "message": "",
        "output_path": None
    }
    
    try:
        logger.info("开始执行横屏转竖屏功能")
        
        video_path = os.path.join(INPUT_VIDEO_DIR, video_filename)
        if not os.path.exists(video_path):
            result["message"] = f"视频文件不存在: {video_path}"
            return result
        
        video_name = os.path.splitext(video_filename)[0]
        output_filename = f"{video_name}_vertical.mp4"
        output_path = os.path.join(VERTICAL_VIDEO_DIR, output_filename)
        
        logger.info(f"转换视频: {video_filename}, 方法: {method}")
        
        # 初始化视频处理器
        video_processor = VideoProcessor()
        
        # 调用转换方法
        success = video_processor.convert_to_vertical(video_path, output_path, method=method)
        
        if success:
            result["success"] = True
            result["output_path"] = output_path
            result["message"] = f"转换完成: {output_filename}"
            logger.info(f"横屏转竖屏完成，输出文件: {output_path}")
        else:
            result["message"] = "转换失败"
            logger.error("横屏转竖屏失败")
        
    except Exception as e:
        logger.exception(f"横屏转竖屏出错: {str(e)}")
        result["message"] = f"横屏转竖屏出错: {str(e)}"
    
    return result

def add_subtitles_to_video(video_filename: str, 
                          transcript_filename: Optional[str] = None) -> Dict[str, Any]:
    """
    为视频添加字幕功能
    
    Args:
        video_filename: 视频文件名
        transcript_filename: 可选的转录文件名，如果不提供则自动查找
        
    Returns:
        包含添加字幕结果的字典
    """
    result = {
        "success": False,
        "message": "",
        "output_path": None
    }
    
    try:
        logger.info("开始执行为视频添加字幕功能")
        
        video_path = os.path.join(INPUT_VIDEO_DIR, video_filename)
        if not os.path.exists(video_path):
            result["message"] = f"视频文件不存在: {video_path}"
            return result
        
        video_name = os.path.splitext(video_filename)[0]
        
        # 如果没有指定转录文件，自动查找
        if not transcript_filename:
            transcript_filename = f"{video_name}_transcript.json"
        
        transcript_path = os.path.join(TRANSCRIPTS_DIR, transcript_filename)
        
        if not os.path.exists(transcript_path):
            result["message"] = f"转录文件不存在: {transcript_path}"
            return result
        
        output_filename = f"{video_name}_with_subtitles.mp4"
        output_path = os.path.join(OUTPUT_VIDEO_DIR, output_filename)
        
        logger.info(f"为视频添加字幕: {video_filename}")
        
        # 初始化视频处理器
        video_processor = VideoProcessor()
        
        # 调用添加字幕方法
        success = video_processor.add_subtitles(video_path, transcript_path, output_path)
        
        if success:
            result["success"] = True
            result["output_path"] = output_path
            result["message"] = f"字幕添加完成: {output_filename}"
            logger.info(f"字幕添加完成，输出文件: {output_path}")
        else:
            result["message"] = "添加字幕失败"
            logger.error("字幕添加失败")
        
    except Exception as e:
        logger.exception(f"添加字幕出错: {str(e)}")
        result["message"] = f"添加字幕出错: {str(e)}"
    
    return result

def get_libraries() -> List[str]:
    """获取所有可用的逻辑库名称"""
    try:
        vkb = VideoKnowledgeBase()
        # 这里需要根据实际情况返回库列表
        # 暂时返回默认库
        return ["default_lib"]
    except:
        return ["default_lib"]

# ==================== 初始化 ====================

# 确保目录存在
ensure_directories()
logger.info("视频智能剪辑工具后端服务初始化完成")

# 导出函数列表
__all__ = [
    'ensure_directories',
    'get_video_files',
    'get_transcript_files',
    'get_rag_files',
    'get_analysis_files',
    'get_video_md5',
    'get_libraries',
    'data_processing',
    'rag_building',
    'rag_search',
    'video_editing',
    'convert_to_vertical',
    'add_subtitles_to_video',
    'INPUT_VIDEO_DIR',
    'OUTPUT_VIDEO_DIR',
    'TRANSCRIPTS_DIR',
    'RAGSCRIPTS_DIR',
    'ANALYSIS_RESULTS_DIR',
    'SLICE_VIDEO_DIR',
    'KEYFRAMES_DIR',
    'VERTICAL_VIDEO_DIR',
    'VIDEO_POOL_DIR'
]