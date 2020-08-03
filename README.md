# multimodal-processing
##数据
ResNet50预处理倒数第二层得到2048维图片向量，每个图片对应文本标签平均26.5个
##模型
两层2048维全连接层 + ArcFace论文方法处理的网络层
##环境
python: 3.7.7  tensorflow: 2.2.0 keras: 2.4.0
