import streamlit as st
import json
from urllib.parse import urlparse
import os

def load_json_data(file_path):
    """从JSON文件加载数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"文件未找到: {file_path}")
        return []
    except json.JSONDecodeError:
        st.error(f"无法解析JSON文件: {file_path}")
        return []

def get_filename_from_url(url):
    """从URL中提取文件名"""
    try:
        path = urlparse(url).path
        return os.path.basename(path)
    except:
        return "unknown_file"

def main():
    st.set_page_config(
        page_title="视频搜索结果可视化",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("视频搜索结果可视化")
    st.markdown("展示视频搜索结果，支持预览和筛选")
    
    # 侧边栏：搜索和筛选
    with st.sidebar:
        st.header("搜索与筛选")
        
        # 文件上传
        uploaded_file = st.file_uploader("上传JSON文件", type=["json"])
        
        # 搜索关键词
        search_term = st.text_input("搜索关键词", "")
        
        # 相似度分数筛选
        min_score = st.slider("最小相似度分数", 0.0, 1.0, 0.0)
        max_score = st.slider("最大相似度分数", 0.0, 1.0, 1.0)
        
        # 视频时长筛选
        time_filter = st.selectbox(
            "视频时长",
            ["全部", "0-10秒", "10-30秒", "30秒-1分钟", "1分钟以上"]
        )
    
    # 加载数据
    if uploaded_file is not None:
        data = json.load(uploaded_file)
    else:
        raise ValueError("请上传JSON文件")
    # 筛选数据
    filtered_data = []
    for item in data:
        # 检查是否为列表中的项
        if isinstance(item, dict):
            # 搜索关键词筛选
            if search_term.lower() not in item.get("search_keyword", "").lower():
                continue
            
            # 相似度分数筛选
            if item.get("sim_score", 0) < min_score or item.get("sim_score", 0) > max_score:
                continue
            
            # 视频时长筛选
            duration = item.get("video_time_duration", "00:00:00")
            try:
                h, m, s = map(int, duration.split(':'))
                total_seconds = h * 3600 + m * 60 + s
                
                if time_filter == "0-10秒" and not (0 <= total_seconds <= 10):
                    continue
                elif time_filter == "10-30秒" and not (10 < total_seconds <= 30):
                    continue
                elif time_filter == "30秒-1分钟" and not (30 < total_seconds <= 60):
                    continue
                elif time_filter == "1分钟以上" and not (total_seconds > 60):
                    continue
            except:
                if time_filter != "全部":
                    continue
            
            filtered_data.append(item)
    
    # 显示统计信息
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"总结果数: {len(data)}")
    with col2:
        st.success(f"筛选后结果数: {len(filtered_data)}")
    
    # 显示视频卡片
    if filtered_data:
        for i, item in enumerate(filtered_data):
            with st.container():
                col2 = st.columns(1)
                # col1, col2 = st.columns([1, 3])
                
                # with col1:
                #     pass
                #     # 显示封面图片
                #     cover_url = item.get("cover_image")
                #     if cover_url:
                #         st.image(cover_url, caption="封面图", use_container_width=True)
                #     else:
                #         st.info("无封面图片")
                
                with col2[0]:
                    # 显示基本信息
                    st.subheader(f"关键词: {item.get('search_keyword', '未知')}")
                    st.subheader(f"问题: {item.get('question', '未知')}")
                    
                    # 显示相似度分数
                    sim_score = item.get("sim_score", 0)
                    st.progress(sim_score)
                    st.markdown(f"<h4>相似度分数: {sim_score:.2f}</h4>", unsafe_allow_html=True)
                    
                    # 显示其他信息
                    col3, col4 = st.columns(2)
                    col3.text(f"来源: {item.get('source', '未知')}")
                    col3.text(f"时长: {item.get('video_time_duration', '未知')}")
                    col4.text(f"分辨率: {item.get('scale', '未知')}")
                    col4.text(f"ID: {item.get('id', '未知')[:8]}...")
                    
                    # 视频播放器
                    video_url = item.get("video_url")
                    if video_url:
                        st.video(video_url)
                    else:
                        st.error("无法获取视频URL")
                    
                    # 下载链接
                    if video_url:
                        file_name = get_filename_from_url(video_url)
                        st.markdown(f"[下载视频]({video_url})", unsafe_allow_html=True)
            
            # 添加分隔线
            if i < len(filtered_data) - 1:
                st.markdown("---")
    else:
        st.warning("没有找到匹配的视频")

if __name__ == "__main__":
    main()    
