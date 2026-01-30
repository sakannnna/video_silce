import os
import json
import logging
import requests
import re
from config import DASHSCOPE_API_KEY, TRANSCRIPTS_DIR
from typing import List, Dict
import dashscope
from dashscope.audio.asr import Transcription

logger = logging.getLogger(__name__)

class SpeechToText:
    """语音识别器，使用Paraformer进行语音转文字"""
    
    def __init__(self):
        """
        初始化语音识别模型或者API的调用设置
        """
        self.api_key = DASHSCOPE_API_KEY
        if not self.api_key:
            logger.warning("未找到 DASHSCOPE_API_KEY，请在 .env 文件中配置")
            print("警告: 未找到 DASHSCOPE_API_KEY，语音识别将无法正常工作")
        
        dashscope.api_key = self.api_key

    def transcribe(self, audio_path):
        """
        将音频文件转录为带时间戳的文本
        参数Args:
            audio_path: 音频文件路径
        返回Returns:
            列表list: 带时间戳的单词/句子列表
                每个元素为字典格式: {"word": str, "start": float, "end": float}
        """
        if not self.api_key:
            print("Error: Missing DASHSCOPE_API_KEY. Cannot transcribe.")
            # 尝试回退到伪造数据（仅用于测试/演示）
            return self._fallback_fake_transcribe(audio_path)

        if not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            return []

        logger.info(f"开始使用 Paraformer (Transcription) 识别音频: {audio_path}")
        
        # 使用 Paraformer v2 模型和 Transcription 接口
        try:
            # 尝试直接传递本地文件路径 (file://...)
            # DashScope Python SDK 在较新版本支持本地文件自动上传
            file_url = f"file://{audio_path}"
            
            # 提交转写任务
            # 注意：Transcription 接口通常是异步的
            task_response = Transcription.async_call(
                model='paraformer-v2',
                file_urls=[file_url]
            )
            
            logger.info(f"转写任务已提交，Task ID: {task_response.output.task_id}")
            
            # 等待任务完成
            # 使用 wait 方法轮询任务状态
            transcription_response = Transcription.wait(task=task_response.output.task_id)
            
            if transcription_response.status_code == 200:
                if transcription_response.output.task_status == 'SUCCEEDED':
                    logger.info("转写任务成功完成")
                    results = []
                    
                    # 解析结果
                    # Transcription 的结果通常在 results 字段中
                    # 每一个 result 对应一个输入文件
                    # 结构通常是: output.results[0].subtask_status == 'SUCCEEDED'
                    # 且 output.results[0].sentences 包含句子列表
                    
                    for result in transcription_response.output.results:
                        if result.subtask_status == 'SUCCEEDED':
                            # 获取 sentences_url 并下载? 或者直接在 sentences 字段中?
                            # DashScope SDK 的 wait 方法通常会返回完整结果
                            sentences = getattr(result, 'sentences', None)
                            if sentences:
                                for sentence in sentences:
                                    text = sentence.get('text', '')
                                    start_time = sentence.get('begin_time', 0) / 1000.0
                                    end_time = sentence.get('end_time', 0) / 1000.0
                                    
                                    results.append({
                                        "word": text,
                                        "start": start_time,
                                        "end": end_time
                                    })
                            elif hasattr(result, 'sentences_url') and result.sentences_url:
                                # 如果结果太大，可能会返回 URL
                                try:
                                    logger.info(f"下载 sentences_url: {result.sentences_url}")
                                    resp = requests.get(result.sentences_url)
                                    resp.raise_for_status()
                                    sentences_data = resp.json()
                                    # sentences_data 可能就是 sentences 列表，或者包含 sentences 字段
                                    # 根据 API 文档，sentences_url 下载的内容通常是一个 JSON，包含 sentences 列表
                                    # 这里假设下载的 JSON 直接是 sentences 列表，或者结构类似
                                    
                                    # 检查是否是字典且包含 sentences
                                    if isinstance(sentences_data, dict) and 'sentences' in sentences_data:
                                        sentences = sentences_data['sentences']
                                    elif isinstance(sentences_data, list):
                                        sentences = sentences_data
                                    else:
                                        sentences = []

                                    for sentence in sentences:
                                        text = sentence.get('text', '')
                                        start_time = sentence.get('begin_time', 0) / 1000.0
                                        end_time = sentence.get('end_time', 0) / 1000.0
                                        
                                        results.append({
                                            "word": text,
                                            "start": start_time,
                                            "end": end_time
                                        })
                                except Exception as e:
                                    logger.error(f"下载或解析 sentences_url 失败: {e}")
                            else:
                                logger.warning("未找到 sentences 或 sentences_url")
                        else:
                            logger.error(f"子任务失败: {result.message}")
                            
                    logger.info(f"识别完成，共获取 {len(results)} 条句子")
                    return self._normalize(results)
                else:
                    logger.error(f"转写任务失败: {transcription_response.output.task_status}")
                    return self._fallback_fake_transcribe(audio_path)
            else:
                logger.error(f"API 调用失败: {transcription_response.code} - {transcription_response.message}")
                return self._fallback_fake_transcribe(audio_path)

        except Exception as e:
            logger.error(f"调用 Paraformer API 失败: {e}")
            print(f"调用 Paraformer API 失败: {e}")
            return self._fallback_fake_transcribe(audio_path)

    def _fallback_fake_transcribe(self, audio_path):
        """回退到原有的伪造逻辑（仅当API不可用时）"""
        logger.warning("使用伪造的转录数据作为回退")
        base = os.path.splitext(os.path.basename(audio_path))[0]
        candidates = [
            os.path.join(TRANSCRIPTS_DIR, f"{base}.json"),
            os.path.join(TRANSCRIPTS_DIR, "text_trans.json"),
        ]
        data = []
        for path in candidates:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except Exception:
                        data = []
                break
        return self._normalize(data)

    def split_by_punctuation(self, items):
        """
        按标点将 ASR 句子切分成更短的片段（用于视频剪辑）
        """
        segments = []
        punctuation_pattern = r'[，。！？；,.!?;]'

        for item in items:
            text = item["word"]
            start = item["start"]
            end = item["end"]

            duration = end - start
            if duration <= 0 or not text:
                continue

            # 按标点切文本（保留顺序）
            parts = re.split(punctuation_pattern, text)
            parts = [p.strip() for p in parts if p.strip()]

            if len(parts) == 1:
                # 没标点，直接返回原句
                segments.append(item)
                continue

            total_chars = sum(len(p) for p in parts)
            current_time = start

            for i, part in enumerate(parts):
                ratio = len(part) / total_chars
                part_duration = duration * ratio

                part_start = current_time
                part_end = end if i == len(parts) - 1 else current_time + part_duration

                segments.append({
                    "word": part,
                    "start": round(part_start, 3),
                    "end": round(part_end, 3)
                })

                current_time = part_end

        return segments

    def _normalize(self, items: List[Dict]) -> List[Dict]:
        result: List[Dict] = []
        fillers = ["嗯", "啊", "那个", "就是", "然后", "呃", "喂喂", "哦"]
        for item in items or []:
            t = item.get("text", item.get("word", ""))
            for f in fillers:
                t = t.replace(f, "")
            t = t.strip()
            try:
                s = float(item.get("start", 0.0))
                e = float(item.get("end", 0.0))
            except Exception:
                continue
            if t and e > s:
                # 按照接口定义，返回 key 为 "word"
                result.append({"word": t, "start": s, "end": e})
        return result
