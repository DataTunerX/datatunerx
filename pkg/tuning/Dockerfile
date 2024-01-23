FROM rayproject/ray271-py39-gpu-llama2-7b-inference:20231220

WORKDIR /tuning

COPY requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .