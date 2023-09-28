from extract_retinal_vessle import extract_vessle
import numpy as np
import streamlit as st
import cv2

with st.sidebar:
    st.markdown('#### 参数调整')
    preprocess=st.container()
    st.markdown("---")
    DoG=st.container()
    st.markdown("---")
    blurring=st.container()
    st.markdown("---")
    shifting=st.container()
    st.markdown("---")
    geometric_mean=st.container()
    st.markdown("---")
    morphological=st.container()
    st.markdown("---")
    binary=st.container()
    st.markdown("---")
    post_processing=st.container()
    st.markdown("---")
    resize=st.container()
    reset_button=st.button('重置参数')

reset_params={
    "fov_threshold":0.1,
    "clahe_clip_limit":2.0,
    "clahe_grid_size":8,
    "DoG_size":7,
    "DoG_sigma":1.5,
    "blurring_sigma":1.0,
    "blurring_alpha":0.5,
    "blurring_rho_sample":1.0,
    "shifting_phi_sample":0.392,
    "geometric_mean_sigma":1.5,
    "geometric_mean_t":1.0,
    "structuring_element_size":3,
    "binary_threshold":0.05,
    "post_processing_gap":10,
    "resize_size":256
}

# 参数需要用户输入
fov_threshold=preprocess.slider(
    'FOV阈值',0.0,1.0,reset_params["fov_threshold"])
clahe_clip_limit=preprocess.slider(
    'CLAHE的对比度限制',0.0,10.0,reset_params["clahe_clip_limit"])
clahe_grid_size=preprocess.number_input(
    'CLAHE的网格大小',min_value=1,max_value=20,value=reset_params["clahe_grid_size"],step=1)
DoG_size=DoG.number_input(
    'DoG滤波器大小',min_value=1,max_value=20,value=reset_params["DoG_size"],step=1)
DoG_sigma=DoG.slider(
    'DoG滤波器标准差',0.0,10.0,reset_params["DoG_sigma"])
blurring_sigma=blurring.slider(
    '初始标准差',0.0,10.0,reset_params["blurring_sigma"])
blurring_alpha=blurring.slider(
    '缩放因子',0.0,1.0,reset_params["blurring_alpha"])
blurring_rho_sample=blurring.slider(
    'rho采样值',0.0,10.0,reset_params["blurring_rho_sample"])
shifting_phi_sample=shifting.slider(
    'phi采样值',0.0,3.14,reset_params["shifting_phi_sample"])
geometric_mean_sigma=geometric_mean.slider(
    '几何平均标准差',0.0,10.0,reset_params["geometric_mean_sigma"])
geometric_mean_t=geometric_mean.slider(
    '几何平均阈值',0.0,10.0,reset_params["geometric_mean_t"])
structuring_element_size=morphological.number_input(
    '形态处理结构元素大小',min_value=1,max_value=20,value=reset_params["structuring_element_size"],step=1)
binary_threshold=binary.slider(
    '二值化阈值',0.0,1.0,reset_params["binary_threshold"])
post_processing_gap=post_processing.number_input(
    '填充间隙的最大大小',min_value=1,max_value=20,value=reset_params["post_processing_gap"],step=1)
resize_size=resize.number_input(
    '重采样图像大小',min_value=0,max_value=1024,value=reset_params["resize_size"],step=1)

# 说明
st.markdown('''
### 视网膜血管分割

算法基于以下论文：
> Li W, Xiao Y, Hu H, Zhu C, Wang H, Liu Z, Sangaiah AK. Retinal Vessel Segmentation Based on B-COSFIRE Filters in Fundus Images. Front Public Health. 2022 Sep 9;10:914973. doi: 10.3389/fpubh.2022.914973. PMID: 36159307; PMCID: PMC9500397.
[https://www.frontiersin.org/articles/10.3389/fpubh.2022.914973/full](https://www.frontiersin.org/articles/10.3389/fpubh.2022.914973/full)
''')
# 上传图片
uploaded_file = st.file_uploader("上传视网膜图片", type=['png', 'jpg', 'jpeg','tif'], accept_multiple_files=False)

if uploaded_file is not None:
    original_image  = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    imagelist, original_size = extract_vessle(original_image,fov_threshold,clahe_clip_limit,clahe_grid_size,DoG_size,DoG_sigma,blurring_sigma,blurring_alpha,blurring_rho_sample,shifting_phi_sample,geometric_mean_sigma,geometric_mean_t,structuring_element_size,binary_threshold,post_processing_gap,resize_size)


    retina=cv2.resize(imagelist[0],(original_size[1],original_size[0]))
    vessel=cv2.resize(imagelist[-2],(original_size[1],original_size[0]))
    combined=cv2.resize(imagelist[-1],(original_size[1],original_size[0]))
    col1,col2,col3=st.columns(3)

    col1.image(retina,caption='视网膜图像')
    col2.image(vessel,caption='血管图像')
    col3.image(combined,caption='标记图像')

