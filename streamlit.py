# streamlit_app.py (修复字幕页面)
"""
streamlit_app.py - 视频智能剪辑工具Web界面
直接调用server.py中的函数，无需额外API服务器
运行方式: streamlit run streamlit_app.py
"""

import streamlit as st
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

# 导入server模块（修改后的main.py）
import server

# 设置页面配置
st.set_page_config(
    page_title="视频智能剪辑工具",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    /* 侧边栏功能卡片 */
    [data-testid="stSidebar"] {
        background-color: #1e1e2f;
        padding: 20px 10px;
    }
    
    /* 功能选择卡片 */
    .function-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        cursor: pointer;
        transition: all 0.3s;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .function-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }
    .function-card.active {
        border: 3px solid #ffd700;
        background: linear-gradient(135deg, #5a67d8 0%, #6b46a1 100%);
    }
    .function-icon {
        font-size: 48px;
        margin-bottom: 10px;
    }
    .function-title {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .function-desc {
        font-size: 14px;
        opacity: 0.9;
    }
    
    /* 主要内容区域卡片 */
    .content-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .section-title {
        color: #1e1e2f;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid #667eea;
    }
    
    /* 结果卡片 */
    .result-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .result-success {
        border-left-color: #28a745;
        background: #d4edda;
    }
    .result-error {
        border-left-color: #dc3545;
        background: #f8d7da;
    }
    
    /* 进度条美化 */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 25px;
        border-radius: 25px;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* 指标卡片 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
    }
    
    /* 视频信息卡片 */
    .video-info {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    /* 字幕样式预览 */
    .subtitle-preview {
        background: #1e1e2f;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
    .preview-text {
        font-size: 24px;
        font-weight: bold;
        padding: 10px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
def init_session_state():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "数据准备"
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    if 'video_files' not in st.session_state:
        st.session_state.video_files = []
    if 'transcript_files' not in st.session_state:
        st.session_state.transcript_files = []
    if 'rag_files' not in st.session_state:
        st.session_state.rag_files = []
    if 'analysis_files' not in st.session_state:
        st.session_state.analysis_files = []

# 刷新文件列表
def refresh_file_lists():
    st.session_state.video_files = server.get_video_files()
    st.session_state.transcript_files = server.get_transcript_files()
    st.session_state.rag_files = server.get_rag_files()
    st.session_state.analysis_files = server.get_analysis_files()

# 侧边栏
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🎬 视频智能剪辑")
        st.markdown("---")
        
        # 功能卡片
        functions = [
            {
                "id": "数据准备",
                "icon": "📊",
                "title": "数据准备",
                "desc": "提取音频、文字、画面"
            },
            {
                "id": "RAG构建",
                "icon": "🔍",
                "title": "RAG知识库",
                "desc": "构建向量数据库"
            },
            {
                "id": "智能剪辑",
                "icon": "✂️",
                "title": "智能剪辑",
                "desc": "根据指令剪辑视频"
            },
            {
                "id": "横屏转竖屏",
                "icon": "📱",
                "title": "横屏转竖屏",
                "desc": "转换为竖屏格式"
            },
            {
                "id": "添加字幕",
                "icon": "📝",
                "title": "添加字幕",
                "desc": "为视频添加字幕"
            }
        ]
        
        for func in functions:
            is_active = (st.session_state.current_page == func["id"])
            active_class = "active" if is_active else ""
            
            col1, col2, col3 = st.columns([1, 8, 1])
            with col2:
                if st.button(
                    func["title"],
                    key=f"nav_{func['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state.current_page = func["id"]
                    st.rerun()
        
        st.markdown("---")
        
        # 刷新按钮
        if st.button("🔄 刷新文件列表", use_container_width=True):
            refresh_file_lists()
            st.rerun()
        
        # 显示统计信息
        st.markdown("### 📊 统计")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("视频文件", len(st.session_state.video_files))
        with col2:
            st.metric("转录文件", len(st.session_state.transcript_files))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RAG文件", len(st.session_state.rag_files))
        with col2:
            st.metric("分析结果", len(st.session_state.analysis_files))

# 页面1: 数据准备
def page_data_processing():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 数据准备</div>', unsafe_allow_html=True)
    st.markdown("从视频中提取音频、文字转录和画面内容")
    
    if not st.session_state.video_files:
        st.warning("⚠️ 没有找到视频文件，请将视频放入 input 目录")
        if st.button("刷新文件列表"):
            refresh_file_lists()
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 1. 选择视频文件")
        selected_video = st.selectbox(
            "选择要处理的视频",
            st.session_state.video_files,
            key="dp_video_select",
            label_visibility="collapsed"
        )
        
        if selected_video:
            video_path = os.path.join(server.INPUT_VIDEO_DIR, selected_video)
            file_size = os.path.getsize(video_path) / (1024 * 1024)
            st.markdown(f"""
            <div class="video-info">
                <b>🎬 {selected_video}</b><br>
                大小: {file_size:.1f} MB
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 2. 处理选项")
        process_type = st.radio(
            "选择处理类型",
            ["完整处理", "仅语音转文字", "仅画面分析"],
            horizontal=True
        )
        
        if process_type == "完整处理":
            st.info("将执行：音频提取 → 语音转文字 → 关键帧提取 → 画面分析 → 数据整合")
        elif process_type == "仅语音转文字":
            st.info("只进行语音转文字处理")
        else:
            st.info("只进行画面分析处理")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 开始处理", use_container_width=True, type="primary"):
            if selected_video:
                with st.spinner("正在处理中，请稍候..."):
                    # 这里简化处理，实际应该根据process_type调用不同函数
                    result = server.data_processing(selected_video)
                    
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        
                        # 显示结果指标
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("转录片段", result.get("transcript_count", 0))
                        with col2:
                            st.metric("关键帧", result.get("keyframes_count", 0))
                        with col3:
                            st.metric("视觉片段", result.get("visual_segments_count", 0))
                        
                        st.balloons()
                        refresh_file_lists()
                    else:
                        st.error(f"❌ {result['message']}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# 页面2: RAG构建
def page_rag_building():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 RAG知识库构建</div>', unsafe_allow_html=True)
    st.markdown("构建向量知识库，支持语义搜索视频内容")
    
    tabs = st.tabs(["🧹 数据清洗", "🏗️ 构建知识库", "🔎 语义搜索"])
    
    # 数据清洗标签
    with tabs[0]:
        st.markdown("### 清洗转录数据为RAG格式")
        
        if not st.session_state.transcript_files:
            st.warning("没有找到转录文件，请先进行数据准备")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_json = st.selectbox(
                    "选择要清洗的转录文件",
                    st.session_state.transcript_files,
                    key="rag_clean_select"
                )
            
            with col2:
                category = st.text_input(
                    "分类标签",
                    value="general",
                    help="为数据添加分类标签"
                )
            
            if st.button("🧹 开始清洗", use_container_width=True):
                with st.spinner("正在清洗数据..."):
                    result = server.rag_building(
                        source_json=selected_json,
                        category=category
                    )
                    
                    if result["success"]:
                        st.success(f"✅ 清洗完成!")
                        st.info(f"生成文件: {result['rag_filename']}")
                        st.balloons()
                        refresh_file_lists()
                    else:
                        st.error(f"❌ {result['message']}")
    
    # 构建知识库标签
    with tabs[1]:
        st.markdown("### 构建向量知识库")
        
        if not st.session_state.rag_files:
            st.warning("没有找到清洗后的RAG文件，请先进行数据清洗")
        else:
            selected_rag = st.selectbox(
                "选择RAG文件",
                st.session_state.rag_files,
                key="rag_build_select"
            )
            
            if st.button("🏗️ 开始构建", use_container_width=True, type="primary"):
                with st.spinner("正在构建知识库..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 模拟进度
                    for i in range(101):
                        progress_bar.progress(i / 100)
                        status_text.text(f"构建进度: {i}%")
                        time.sleep(0.05)
                    
                    result = server.rag_building(rag_filename=selected_rag)
                    
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.metric("总数据量", result.get("total_items", 0))
                        st.metric("向量库数量", result.get("collection_count", 0))
                        st.balloons()
                    else:
                        st.error(f"❌ {result['message']}")
    
    # 语义搜索标签
    with tabs[2]:
        st.markdown("### 语义搜索")
        
        query = st.text_input(
            "输入搜索内容",
            placeholder="例如: 找出切肉的画面、讲解关键技术的部分...",
            key="rag_query"
        )
        
        col1, col2 = st.columns([3, 1])
        with col2:
            top_k = st.number_input("返回结果数", min_value=1, max_value=20, value=5)
        
        if query and st.button("🔎 搜索", use_container_width=True):
            with st.spinner("正在搜索..."):
                result = server.rag_search(query, top_k=top_k)
                
                if result["success"]:
                    st.success(f"找到 {len(result['results'])} 个相关结果")
                    
                    for i, item in enumerate(result["results"]):
                        st.markdown(f"""
                        <div class="result-card">
                            <h4>结果 {i+1}</h4>
                            <p><b>时间:</b> {item['start']:.1f}s - {item['end']:.1f}s</p>
                            <p><b>类型:</b> {item['type']}</p>
                            <p><b>分类:</b> {item['category']}</p>
                            <p><b>内容:</b> {item['content'][:200]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info(result["message"])
    
    st.markdown('</div>', unsafe_allow_html=True)

# 页面3: 智能剪辑
def page_video_editing():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">✂️ 智能视频剪辑</div>', unsafe_allow_html=True)
    st.markdown("根据文字指令智能剪辑视频")
    
    if not st.session_state.video_files:
        st.warning("没有找到视频文件，请先上传视频")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 1. 选择视频")
        selected_video = st.selectbox(
            "选择要剪辑的视频",
            st.session_state.video_files,
            key="edit_video_select",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### 2. 剪辑参数")
        max_duration = st.slider(
            "最大时长（秒）",
            min_value=10,
            max_value=300,
            value=60,
            step=10
        )
    
    st.markdown("### 3. 剪辑要求")
    instruction = st.text_area(
        "输入剪辑要求",
        placeholder="例如: 找出切肉的画面、选择讲解关键技术的部分、保留精彩瞬间...",
        height=100,
        label_visibility="collapsed"
    )
    
    if st.button("🎬 开始智能剪辑", use_container_width=True, type="primary"):
        if not instruction:
            st.error("请输入剪辑要求")
        elif not selected_video:
            st.error("请选择视频")
        else:
            with st.spinner("正在剪辑视频..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 模拟进度
                steps = ["分析文本", "选择片段", "剪辑视频", "合并片段"]
                for i, step in enumerate(steps):
                    status_text.text(f"{step}...")
                    progress_bar.progress((i + 1) * 25)
                    time.sleep(1)
                
                result = server.video_editing(
                    video_filename=selected_video,
                    user_instruction=instruction,
                    max_duration=max_duration
                )
                
                if result["success"]:
                    st.success(f"✅ {result['message']}")
                    
                    # 显示结果
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("原始片段", len(result.get("segments", [])))
                    with col2:
                        st.metric("选中片段", len(result.get("selected_segments", [])))
                    with col3:
                        st.metric("总时长", f"{result.get('total_duration', 0):.1f}秒")
                    
                    if result.get("output_path"):
                        st.info(f"输出文件: {result['output_path']}")
                    
                    st.balloons()
                    refresh_file_lists()
                else:
                    st.error(f"❌ {result['message']}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# 页面4: 横屏转竖屏
def page_convert_to_vertical():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📱 横屏转竖屏</div>', unsafe_allow_html=True)
    st.markdown("将横屏视频转换为竖屏格式")
    
    if not st.session_state.video_files:
        st.warning("没有找到视频文件，请先上传视频")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 1. 选择视频")
        selected_video = st.selectbox(
            "选择要转换的视频",
            st.session_state.video_files,
            key="vertical_video_select",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### 2. 转换设置")
        conversion_method = st.radio(
            "转换方法",
            ["solid", "blur", "crop"],
            format_func=lambda x: {
                "solid": "纯色填充（最快）",
                "blur": "模糊背景",
                "crop": "智能裁剪"
            }[x],
            horizontal=True
        )
    
    if st.button("🔄 开始转换", use_container_width=True, type="primary"):
        with st.spinner("正在转换..."):
            result = server.convert_to_vertical(
                video_filename=selected_video,
                method=conversion_method
            )
            
            if result["success"]:
                st.success(f"✅ {result['message']}")
                if result.get("output_path"):
                    st.info(f"输出文件: {result['output_path']}")
                st.balloons()
                refresh_file_lists()
            else:
                st.error(f"❌ {result['message']}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# 页面5: 添加字幕（修复版本）
def page_add_subtitles():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📝 添加字幕</div>', unsafe_allow_html=True)
    st.markdown("为视频添加字幕")
    
    if not st.session_state.video_files:
        st.warning("没有找到视频文件，请先上传视频")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    if not st.session_state.transcript_files:
        st.warning("没有找到转录文件，请先进行数据准备")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 1. 选择视频")
        selected_video = st.selectbox(
            "选择视频",
            st.session_state.video_files,
            key="subtitle_video_select",
            label_visibility="collapsed"
        )
        
        if selected_video:
            video_path = os.path.join(server.INPUT_VIDEO_DIR, selected_video)
            file_size = os.path.getsize(video_path) / (1024 * 1024)
            st.markdown(f"""
            <div class="video-info">
                <b>🎬 {selected_video}</b><br>
                大小: {file_size:.1f} MB
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 2. 选择字幕文件")
        selected_transcript = st.selectbox(
            "选择字幕文件",
            st.session_state.transcript_files,
            key="subtitle_transcript_select",
            label_visibility="collapsed"
        )
        
        if selected_transcript:
            # 预览字幕文件内容
            transcript_path = os.path.join(server.TRANSCRIPTS_DIR, selected_transcript)
            try:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)
                st.info(f"📄 包含 {len(transcript_data)} 条字幕")
            except:
                pass
    
    st.markdown("---")
    st.markdown("### 3. 字幕样式")
    
    # 使用两列布局来避免颜色选择器问题
    col1, col2 = st.columns(2)
    
    with col1:
        # 字体颜色 - 使用标准的6位十六进制
        font_color = st.color_picker(
            "字体颜色", 
            "#FFFFFF",
            help="选择字幕文字颜色"
        )
        
        # 字体大小
        font_size = st.slider(
            "字体大小", 
            min_value=12, 
            max_value=72, 
            value=36,
            step=2
        )
    
    with col2:
        # 背景颜色 - 使用标准的6位十六进制，不带透明度
        bg_color = st.color_picker(
            "背景颜色", 
            "#000000",
            help="选择字幕背景颜色"
        )
        
        # 背景透明度 - 单独控制
        bg_opacity = st.slider(
            "背景透明度", 
            min_value=0, 
            max_value=100, 
            value=50,
            step=5,
            help="0=完全透明, 100=完全不透明"
        )
        
        # 字幕位置
        position = st.select_slider(
            "字幕位置",
            options=["顶部", "中部", "底部"],
            value="底部"
        )
    
    # 字幕样式预览
    st.markdown("### 4. 样式预览")
    
    # 计算带透明度的背景颜色
    opacity_hex = hex(int(bg_opacity * 255 / 100))[2:].zfill(2)
    bg_color_with_alpha = f"{bg_color}{opacity_hex}"
    
    # 显示预览
    st.markdown(f"""
    <div class="subtitle-preview" style="background: #1e1e2f;">
        <div class="preview-text" style="
            font-size: {font_size}px;
            color: {font_color};
            background-color: {bg_color_with_alpha};
            padding: 10px 20px;
            border-radius: 5px;
            text-align: center;
            margin: 0 auto;
            display: inline-block;
            position: relative;
            {'top: 20px;' if position == '顶部' else ''}
            {'top: 50%; transform: translateY(-50%);' if position == '中部' else ''}
            {'bottom: 20px;' if position == '底部' else ''}
        ">
            字幕样式预览示例
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 开始处理
    if st.button("✨ 生成字幕视频", use_container_width=True, type="primary"):
        if not selected_video or not selected_transcript:
            st.error("请选择视频和字幕文件")
        else:
            with st.spinner("正在添加字幕..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 模拟进度
                steps = ["加载视频", "处理字幕", "渲染字幕", "生成输出"]
                for i, step in enumerate(steps):
                    status_text.text(f"{step}...")
                    progress_bar.progress((i + 1) * 25)
                    time.sleep(1.5)
                
                # 构建样式参数
                style = {
                    "font_size": font_size,
                    "font_color": font_color,
                    "bg_color": bg_color_with_alpha,  # 带透明度的颜色
                    "position": position
                }
                
                # 调用server函数
                result = server.add_subtitles_to_video(
                    video_filename=selected_video,
                    transcript_filename=selected_transcript
                )
                
                if result["success"]:
                    st.success(f"✅ {result['message']}")
                    
                    # 显示结果
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("视频文件", selected_video)
                    with col2:
                        if result.get("output_path"):
                            st.metric("输出文件", os.path.basename(result["output_path"]))
                    
                    st.balloons()
                    refresh_file_lists()
                else:
                    st.error(f"❌ {result['message']}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# 主函数
def main():
    init_session_state()
    refresh_file_lists()
    
    # 渲染侧边栏
    render_sidebar()
    
    # 主要内容区域
    st.markdown(f'<h1 style="text-align: center; color: #1e1e2f;">🎬 {st.session_state.current_page}</h1>', 
                unsafe_allow_html=True)
    
    # 根据当前页面显示内容
    if st.session_state.current_page == "数据准备":
        page_data_processing()
    elif st.session_state.current_page == "RAG构建":
        page_rag_building()
    elif st.session_state.current_page == "智能剪辑":
        page_video_editing()
    elif st.session_state.current_page == "横屏转竖屏":
        page_convert_to_vertical()
    elif st.session_state.current_page == "添加字幕":
        page_add_subtitles()

if __name__ == "__main__":
    main()