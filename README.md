本项目为基于注意力机制的人脸图像修复展示，大体分为前后端两个部分。

后端是由Python，结合Numpy及Pytorch编写的修复模型。

前端逻辑通过js书写，界面通过React配合MUI组件库设计。

前后端交互通过Flask实现。

#### 运行

在运行之前，我们需要准备模型需要的依赖库及前端所需的组件库。

可以通过pip或者conda虚拟环境安装，进入后端项目主目录inpaint，运行如下命令：

```python
conda create -n pytorch python=3.7
conda activate pytorch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install numpy PIL opencv glob einops
conda install pyqt5
```

随后运行界面程序

```python
python inpaint_server.py
```

在项目主目录安装前端依赖，运行一下命令

```bash
yarn install
yarn start
```

#### 文件说明

```
├─inpaint # 后端 
    ├─inpaint_server # 前后端对接
├─public # 公共资源文件
└─src # 前端
```

#### 效果展示

![](assets/20211205120955.gif)
