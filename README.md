# 视网膜血管分割

## 说明
----            

算法基于以下论文：
> Li W, Xiao Y, Hu H, Zhu C, Wang H, Liu Z, Sangaiah AK. Retinal Vessel Segmentation Based on B-COSFIRE Filters in Fundus Images. Front Public Health. 2022 Sep 9;10:914973. doi: 10.3389/fpubh.2022.914973. PMID: 36159307; PMCID: PMC9500397.
[https://www.frontiersin.org/articles/10.3389/fpubh.2022.914973/full](https://www.frontiersin.org/articles/10.3389/fpubh.2022.914973/full)
            
算法流程：
* 使用对比度受限的自适应直方图均衡（CLAHE）进行对比度增强。
* 对原始RGB图像的CIELab版本的亮度平面进行阈值处理，以产生一个掩码。
* 应用B-COSFIRE滤波器和形态学滤波器来检测血管并去除噪音。
* 使用二值阈值进行血管分割。
* 通过后处理消除未连接的非血管像素，以获得最终的分割图。
            
本程序试图仿照论文中的算法，但是由于作者没有公开代码，因此本程序的实现与论文中的算法存在差距。（比论文示例差好多啊😂）

----

## 运行

浏览器打开[https://retinalvessel.streamlit.app/](https://retinalvessel.streamlit.app/) 上传视网膜照片即可运行。可以手动调整各个参数，以观察参数对结果的影响。