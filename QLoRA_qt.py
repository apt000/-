from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

# 配置参数
model_name = "meta-llama/Llama-2-7b-chat-hf"  # 本地或在线的 Llama2-7B-chat 模型
load_in_4bit = True  # 启用 4-bit 量化
target_modules = ["q_proj", "v_proj"]  # LoRA 作用的模块
lora_r = 8
lora_alpha = 32
lora_dropout = 0.1

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载基础模型并启用量化
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # 自动将模型分配到 GPU
    load_in_4bit=load_in_4bit,
    # load_in_8bit=True,
    quantization_config=bnb.QuantizationConfig(load_in_4bit=True)
)

# 配置 LoRA
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=target_modules,
    task_type="CAUSAL_LM"  # 自回归语言模型任务
)

# 将 LoRA 模型集成到基础模型中
model = get_peft_model(model, lora_config)

# 打印模型信息
print(model)

# 示例输入数据
input_texts = [
    "What are the benefits of using QLoRA for large language models?",
    "Explain how LoRA helps in fine-tuning large models with minimal resources."
]
inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to("cuda")

# 微调或推理示例
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7
)

# 解码输出
decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
for i, output in enumerate(decoded_outputs):
    print(f"Input {i + 1}:\n{input_texts[i]}\n\nResponse:\n{output}\n")
# 保存模型
model.save_pretrained("./quantized_llama2_7b_chat_lora")
