import logging

logger = logging.getLogger(__name__)

def merge_audio_visual_data(transcript, visual_segments):
    """
    融合语音转录和视觉识别结果
    
    Args:
        transcript (list): 语音转录列表 [{'word': str, 'start': float, 'end': float}]
        visual_segments (list): 视觉识别列表 [{'text': str, 'start': float, 'end': float}]
        
    Returns:
        list: 融合后的结构化数据
    """
    merged_data = []
    current_id = 1
    
    # 确保按时间排序
    transcript.sort(key=lambda x: x.get('start', 0))
    visual_segments.sort(key=lambda x: x.get('start', 0))
    
    last_speech_end = 0.0
    
    # 为了防止同一个视觉片段被多次使用，或者为了优化性能，
    # 我们可以复制一份 visual_segments，或者标记已使用的。
    # 这里简单起见，每次遍历查找。数据量不大。
    
    for i, speech in enumerate(transcript):
        speech_start = speech.get('start', 0)
        speech_end = speech.get('end', 0)
        speech_text = speech.get('word', '')
        
        # 1. 检查前面的空隙 (Gap Filling)
        # 如果语音句子之间有超过 3 秒的空隙
        if speech_start - last_speech_end > 3.0:
            gap_start = last_speech_end
            gap_end = speech_start
            
            # 查找落在这个空隙里的图片描述
            # 判定标准：视觉片段的开始时间在空隙内
            gap_visuals = []
            for v in visual_segments:
                v_start = v.get('start', 0)
                if gap_start <= v_start < gap_end:
                    # 提取纯文本描述
                    desc = v.get('text', '').replace("[视觉画面: ", "").replace("]", "")
                    if desc:
                        gap_visuals.append(desc)
            
            # 如果有视觉描述，插入 silent_action
            if gap_visuals:
                merged_data.append({
                    "id": current_id,
                    "time_range": [gap_start, gap_end],
                    "type": "silent_action",
                    "content": "[静默操作]",
                    "visual_context": "画面显示：" + " ".join(gap_visuals)
                })
                current_id += 1
        
        # 2. 处理当前语音句子 (Visual Mapping)
        # 查找每一张图的时间戳落在当前语音句子的范围内
        speech_visuals = []
        for v in visual_segments:
            v_start = v.get('start', 0)
            # 宽松匹配：只要在句子时间范围内
            if speech_start <= v_start <= speech_end:
                desc = v.get('text', '').replace("[视觉画面: ", "").replace("]", "")
                if desc:
                    speech_visuals.append(desc)
        
        # 构造 speech 对象
        visual_context_str = ""
        if speech_visuals:
            visual_context_str = "画面展示：" + " ".join(speech_visuals)
            
        merged_data.append({
            "id": current_id,
            "time_range": [speech_start, speech_end],
            "type": "speech",
            "content": speech_text,
            "visual_context": visual_context_str
        })
        current_id += 1
        
        last_speech_end = speech_end
        
    return merged_data
