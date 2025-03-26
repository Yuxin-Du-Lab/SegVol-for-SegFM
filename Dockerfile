# 使用官方Miniconda镜像作为基础
FROM continuumio/miniconda3

# 设置工作目录
WORKDIR /workspace

# 导入必要的GPG密钥
RUN apt-get update && apt-get install -y gnupg && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32 && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 871920D1991BC93C

# 创建并配置apt源
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-backports main restricted universe multiverse" >> /etc/apt/sources.list 


RUN echo /etc/apt/sources.list

# 更新包列表
RUN apt-get update

# 安装gcc
RUN apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# 配置conda使用清华镜像源
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --set show_channel_urls yes

# 复制环境文件
COPY environment.yml .

# 创建Conda环境
RUN conda env create -f environment.yml

# 设置Shell以使用conda环境
SHELL ["conda", "run", "-n", "segvol", "/bin/bash", "-c"]

# 复制你的代码
COPY . /workspace/

# 设置容器启动时运行的命令
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "segvol", "sh", "infer_case.sh"]

# docker container run --gpus "device=0" -m 8G --name segvol --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ segvol:e2000
  

# docker build -t segvol:e2000 .

# docker save segvol:e2000 | 'C:\Program Files\7-Zip\7z.exe' a -tgzip segvol_e2000.tar.gz
# docker save segvol:e2000 | & "C:\Program Files\7-Zip\7z.exe" a -tgzip segvol_e2000.tar.gz -si