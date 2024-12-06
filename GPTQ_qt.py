import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer, GenerationConfig, pipeline

# 模型名称和路径
model_name = "Llama_2_7B_chat"
out_dir = "Llama_2_7B_chat_qt"

# 量化配置设置
quantize_config = BaseQuantizeConfig(
    bits=16,  
    group_size=128,  
    damp_percent=0.01,  
    desc_act=False  
)

# 加载模型和分词器
model = AutoGPTQForCausalLM.from_pretrained(model_name, quantize_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 准备校准数据集
from datasets import load_dataset

n_samples = 1024
data = load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split=f"train(:{n_samples*5})")
tokenized_data = tokenizer("\n\n".join(data['text']), return_tensors='pt')

# 格式化数据样本
import random

examples_ids = []
for _ in range(n_samples):
    i = random.randint(0, tokenized_data.input_ids.shape[1] - tokenizer.model_max_length - 1)
    j = i + tokenizer.model_max_length
    input_ids = tokenized_data.input_ids[:, i:j]
    attention_mask = torch.ones_like(input_ids)
    examples_ids.append({'input_ids': input_ids, 'attention_mask': attention_mask})

# 执行量化
model.quantize(
    examples_ids,
    batch_size=1,
    use_triton=True
)

# 保存量化后的模型和分词器
model.save_quantized(out_dir, use_safetensors=True)
tokenizer.save_pretrained(out_dir)

# 加载量化后的模型进行测试
quantized_model_name = out_dir
tokenizer = AutoTokenizer.from_pretrained(quantized_model_name)
model = AutoGPTQForCausalLM.from_quantized(quantized_model_name, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")

# 设置生成配置
generation_config = GenerationConfig.from_pretrained(quantized_model_name)
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.0001
generation_config.top_p = 0.95
generation_config.do_sample = True
generation_config.repetition_penalty = 1.15

# 创建文本生成管道
text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    generation_config=generation_config
)

# 输入文本进行生成
input_text = "你好，今天天气如何？"
result = text_pipeline(input_text)[0]['generated_text']
print(result)
