## YOLOPv2-YOLOv8-NCNN-Android
闲来无事写的一个简单的安卓项目，将YOLOPv2和YOLOv8的模型用NCNN运行，用YOLOv8的目标检测替换了YOLOPv2中的目标检测部分，并写成了c++的多线程保证实时性（所以就不计算帧率了，没意义），骁龙8+手机模型计算耗时和效果图如下（来自手机录屏）：
![Screenrecorder-2024-10-02-19-00-40-691 00-01-20 20241003-122219430](https://github.com/user-attachments/assets/e4214cb5-c461-459c-b7c9-ea2333b8c37e)

添加了几个可选功能：
1、可选择的三个任务的计算与绘制：目标检测、车道线识别、可行驶区域识别
![Screenrecorder-2024-10-02-19-00-40-691 00-01-29 20241003-113620681 (1)(1)](https://github.com/user-attachments/assets/658ed16f-48d0-44b7-a46b-9860d9586d13)

2、一个Zoom放大工具条（由于手机摄像头广角太大，用初始画面实际上体现不出来实际距离，所以写的一个简单的放大图像功能，当然这也是为后续功能铺路）
![Screenrecorder-2024-10-02-19-00-40-691 00-00-45 20241003-113526658 (1)(1)](https://github.com/user-attachments/assets/b133db8d-d53d-4cd0-884c-f0471b1c010d)

3、CPU和GPU切换（建议首次运行后马上点开GPU,保证性能）

项目工程里面给了安卓实现

### 目前问题
1. 速度还行，但是不太稳定，测试工具为骁龙8+芯片的手机，yolopv2耗时在50-70ms左右、yolov8n的耗时在20-50不等，做过测试，发热不明显，耗电一般，后面考虑加上bytetrack（有空直接做个辅助驾驶软件吧）
2. 安卓工程并没有做尺寸调整，固定yolopv2的输入为320，而yolov8为640

### 安卓结果
我也导出了APP，给大家下载玩玩: （建议首次运行后马上点开GPU,保证性能）
链接: https://pan.baidu.com/s/1R6TnjWUQuDW3sgJyeBZRjQ?pwd=ykfu 提取码: ykfu

## screenshot
![Screenrecorder-2024-10-02-19-15-32-51 00-05-19 20241003-112756347](https://github.com/user-attachments/assets/8ec0351e-b875-41b2-9c9d-7c09ea2c234d)
缩放功能：![Screenrecorder-2024-10-02-19-00-40-691 00-00-45 20241003-113452820](https://github.com/user-attachments/assets/e6b63636-e682-47a0-8aa9-0afcc30f8eda)
![Screenrecorder-2024-10-02-19-00-40-691 00-00-46 20241003-113532468](https://github.com/user-attachments/assets/cb705f3a-e01d-4a1f-b8d4-92746058148a)
任务选择：![Screenrecorder-2024-10-02-19-00-40-691 00-01-31 20241003-113629142](https://github.com/user-attachments/assets/fcae0434-71ad-40b9-b992-b79f661d516a)
![Screenrecorder-2024-10-02-19-00-40-691 00-01-34 20241003-113635878](https://github.com/user-attachments/assets/a3c39ab3-65be-4bf1-b6e7-e5ccee8af20c)
![Screenrecorder-2024-10-02-19-00-40-691 00-01-41 20241003-113647245](https://github.com/user-attachments/assets/07c1e059-78eb-4972-9da6-a0dc0137a05c)
![Screenrecorder-2024-10-02-19-00-40-691 00-03-05 20241003-113732575](https://github.com/user-attachments/assets/f1c7ce6d-9dad-4432-aef8-f481bf23cf0d)


## reference  

https://github.com/FeiGeChuanShu/YOLOPv2-ncnn

https://github.com/FeiGeChuanShu/ncnn-android-yolov8

https://github.com/CAIC-AD/YOLOPv2

https://github.com/Tencent/ncnn

## If this helped you, don't forget to Star 🌟 the repo.
