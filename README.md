# 大模型量化方法 && Safetensors
## GPTQ
GPTQ是一种针对大规模预训练Transformer模型的精确训练后量化算法，旨在降低模型大小和计算需求，保持高准确度和推理速度。  
通过最小化量化引入的输出误差，在不重新训练模型的情况下，将大模型权重量化到低比特，同时尽可能保持模型的性能。  
[code](https://github.com/apt000/Quantization_method/blob/main/GPTQ_qt.py)  
### 基本步骤  
1.收集校准数据：从训练数据或相关数据集中抽取一小部分样本，作为校准数据。  
2.逐层处理：对模型的每一层进行独立量化，避免全局优化的复杂度。  
3.最小化输出误差：对于每一层，寻找最佳的量化权重，使得在校准数据上的输出误差最小。  
4.更新权重：将量化后的权重替换原始权重。  

## QLoRA
QLoRA 的核心是将大模型的微调过程与量化技术相结合，先将预训练语言模型量化到 4 位精度，再通过低秩适配器（Lora）对量化后的模型进行微调，从而显著减少内存使用量，同时保持较好的性能。  
### 基本步骤  
1.模型量化：采用 4 位 Normal Float（NF4）量化方式，一种适用于正态分布数据的信息理论上最佳量化数据类型，可更好地适应预训练神经网络权重的零中心化正态分布，减少信息损失  
2.低秩适配器（Lora）微调：  1）添加 Lora：在保持大部分原始模型权重固定的情况下，仅优化一小部分可训练的参数，即适配器参数。2）进行微调训练：在训练过程中，梯度通过冻结的 4 位量化预训练语言模型反向传播到低秩适配器中，更新适配器参数以优化损失函数，从而实现对模型的微调。引入低秩矩阵进行调整，进一步降低内存使用并提高训练效率。  
[code](https://github.com/apt000/Quantization_method/blob/main/QLoRA_qt.py)  
## Llama.cpp 
[参考文章](https://blog.csdn.net/m0_65555479/article/details/140949674?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522cc40aafafa3e06f1f630e5d08906bf9e%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=cc40aafafa3e06f1f630e5d08906bf9e&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-140949674-null-null.142^v100^pc_search_result_base6&utm_term=Llama.cpp%E9%87%8F%E5%8C%96&spm=1018.2226.3001.4187)
### 编译Llama.cpp
```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```
### 转换模型权重为 gguf 格式
```
python3 convert-to-gguf.py \
    --model ./llama2-7b-chat \
    --output llama2-7b-chat.gguf \
    --model-size 7B
```
### 量化模型
`./quantize convert-to-gguf.py  --model llama2-7b-chat.gguf  --output llama2-7b-chat-q4_0.gguf  q4_0`
### 使用量化后的模型
`./main -m llama2-7b-chat-q4_0.gguf -p "Explain quantum physics in simple terms." `

# Safetensors
Safetensor 是由 Hugging Face 开发的一种用于安全存储和高效加载张量数据的文件格式及相关技术。  
Safetensor使用特殊的数据结构保证了数据的安全性。采用特定的二进制格式，文件结构包含头部、张量元数据和张量数据。头部记录文件的元数据，如格式版本、数据长度等；张量元数据描述每个张量的名称、数据类型、形状等信息；张量数据则是实际存储的紧凑二进制格式的张量数值。这种结构使得数据的存储和读取更加有序和高效，并且使用校验和机制来防止序列化张量在存储或传输过程中出现损坏，还能防止 DOS 攻击，安全性高。  
Safetensor还针对速度进行了优化，结合高效的序列化和压缩算法，减少大型张量的大小。此外，支持内存映射，文件的布局设计使得读取特定张量时无需加载整个文件，进一步提升了性能，在处理大规模模型时优势明显。  
同时Safetensor的兼容性比普通的tensor格式的跨平台兼容性更好。  
`pip install safetensors`  
**保存为safetensors格式**  
```
import torch
from safetensors.torch import save_file

tensors = {
    "embedding": torch.zeros((2, 2)),
    "attention": torch.zeros((2, 3))
}
save_file(tensors, "model.safetensors")
```
**加载safetensors格式**  
```
from safetensors import safe_open

tensors = {}
with safe_open("model.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
```
**惰性加载**  
可以在不加载整个文件的情况下查看文件的信息或者只加载文件中的部分张量而不是所有张量。  
```
from safetensors import safe_open

tensors = {}
with safe_open("model.safetensors", framework="pt", device=0) as f:
    tensor_slice = f.get_slice("embedding")
    vocab_size, hidden_dim = tensor_slice.get_shape()
    tensor = tensor_slice[:, :hidden_dim]
```
## safetensors的三个组成部分  
**header**：前面的 8 bytes是一个无符号的整数，表示 header 占的字节数，存储了格式版本相关的字节。  
**N bytes**：是一个UTF-8编码JSON字符串，存储 header 的内容，里面为模型权重的元数据信息。  
**rest of file**：存储模型权重 tensor 的值。  
![image](https://github.com/user-attachments/assets/215fed37-ec5d-4fa3-8aea-fbbd015a6374)


