import os
import sys
from moviepy import VideoFileClip, concatenate_videoclips

# 将项目根目录添加到系统路径，以便导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import INPUT_VIDEO_DIR, OUTPUT_VIDEO_DIR
except ImportError:
    # 尝试直接导入（针对不同运行环境的兼容处理）
    try:
        import config
        INPUT_VIDEO_DIR = config.INPUT_VIDEO_DIR
        OUTPUT_VIDEO_DIR = config.OUTPUT_VIDEO_DIR
    except ImportError as e:
        print(f"Warning: Import failed: {e}")
        # 设置默认路径
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        INPUT_VIDEO_DIR = os.path.join(base_dir, "data", "input_videos")
        OUTPUT_VIDEO_DIR = os.path.join(base_dir, "data", "output_videos")

class VideoProcessor:
    """视频处理器，负责根据分析结果剪辑视频"""
    
    def __init__(self):
        """初始化视频处理器"""
        pass
    
    def select_key_clips(self, analyzed_segments, max_duration=300):
        """
        根据评分选择关键片段
        参数Args:
            analyzed_segments: 分析后的片段列表，每个元素包含 start_time, end_time, score, reason
            max_duration: 最大视频时长（秒），默认为5分钟
        返回Returns:
            选择的关键片段列表
        """
        # 按评分降序排序
        sorted_segments = sorted(analyzed_segments, key=lambda x: x.get('score', 0), reverse=True)
        
        selected_clips = []
        total_duration = 0
        
        for segment in sorted_segments:
            segment_duration = segment['end_time'] - segment['start_time']
            # 确保片段时长合理，且总时长不超过限制
            if segment_duration > 0 and total_duration + segment_duration <= max_duration:
                selected_clips.append(segment)
                total_duration += segment_duration
        
        # 按时间顺序排序，确保视频连贯性
        selected_clips.sort(key=lambda x: x['start_time'])
        
        return selected_clips
    
    def combine_clips(self, video_path, selected_segments, output_filename="output_video.mp4"):
        """
        组合片段并生成最终视频
        参数Args:
            video_path: 原始视频路径
            selected_segments: 选择的关键片段列表
            output_filename: 输出视频文件名
        返回Returns:
            输出视频路径
        """
        try:
            # 加载原始视频
            video = VideoFileClip(video_path)
            
            # 提取片段
            clips = []
            for segment in selected_segments:
                # 容错处理：确保时间戳在视频范围内
                start_time = max(0, segment['start_time'])
                end_time = min(video.duration, segment['end_time'])
                
                if end_time > start_time:
                    clip = video.subclip(start_time, end_time)
                    clips.append(clip)
            
            if not clips:
                print("No valid clips to combine.")
                return None
            
            # 组合片段
            final_clip = concatenate_videoclips(clips)
            
            # 生成输出路径
            output_path = os.path.join(OUTPUT_VIDEO_DIR, output_filename)
            
            # 导出视频
            # 设置合理的编码参数，确保音画同步
            final_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                fps=30,
                preset="medium",
                threads=4
            )
            
            # 关闭视频文件
            video.close()
            final_clip.close()
            
            print(f"Video successfully generated: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return None
    
    def process_video(self, video_path, analyzed_segments, max_duration=300, output_filename="output_video.mp4"):
        """
        完整的视频处理流程
        参数Args:
            video_path: 原始视频路径
            analyzed_segments: 分析后的片段列表
            max_duration: 最大视频时长（秒）
            output_filename: 输出视频文件名
        返回Returns:
            输出视频路径
        """
        # 选择关键片段
        selected_segments = self.select_key_clips(analyzed_segments, max_duration)
        
        if not selected_segments:
            print("No segments selected for processing.")
            return None
        
        # 组合片段并生成视频
        output_path = self.combine_clips(video_path, selected_segments, output_filename)
        
        return output_path

# 测试代码
if __name__ == "__main__":
    # 模拟分析结果
    mock_analyzed_segments = [
        {
            "start_time": 10.5,
            "end_time": 20.0,
            "score": 9.5,
            "reason": "核心概念解释"
        },
        {
            "start_time": 30.0,
            "end_time": 45.5,
            "score": 8.7,
            "reason": "重要功能展示"
        },
        {
            "start_time": 60.0,
            "end_time": 75.0,
            "score": 9.0,
            "reason": "案例分析"
        }
    ]
    
    # 测试视频路径（需要根据实际情况修改）
    test_video_path = os.path.join(INPUT_VIDEO_DIR, "test_video.mp4")
    
    processor = VideoProcessor()
    
    # 检查测试视频是否存在
    if os.path.exists(test_video_path):
        output_path = processor.process_video(test_video_path, mock_analyzed_segments)
        print(f"Output video: {output_path}")
    else:
        print(f"Test video not found: {test_video_path}")
        print("Please add a test video to the input_videos directory.")
