# multimodal-processing
## 数据
ResNet50预处理倒数第二层得到2048维图片向量，每个图片对应文本标签平均26.5个
## 图片预处理
ResNet50_vec.py:  调用ResNet50对图片的预处理,得到每个图片的2048维向量.所有结果保存在 "image-data\images.npz"路径下
## 模型
train.py:两层dense layer, 输出激活函数sigmoid, loss:binary_crossentropy

ArcFace.py: 一层普通dense layer + 一层使用论文[1]中方法的arcface layer, 输出激活函数:sigmoid, loss:binary_crossentropy
## 环境
python: 3.7.7  tensorflow: 2.2.0 keras: 2.4.0
## references:
[1] Deng J, Guo J, Xue N, et al. ArcFace: Additive Angular Margin Loss for Deep Face Recognition[J]. arXiv: Computer Vision and Pattern Recognition, 2018.
