# pyttsdata

一个用于生成适用于qwen tts 3的训练数据集的脚本程序
基本流程 输入 -> 识别 -> 删除过短标记 -> 时间戳转换 -> 切分 -> 索引生成

模型默认位于models/
输出默认位于output/

### 如何使用
```bash
git clone https://github.com/Liescake/pyttsdata.git
pip install -r requirements.txt
cd pyttsdata
python -m main.py
```
建议使用虚拟环境运行

### 计划
- [ ] 降噪
- [ ] 终端支持