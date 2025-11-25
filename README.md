# Fine-Tuning LLaMA 2 7B Chat with QLoRA on 5k Instruction Dataset

This project fine-tunes the `NousResearch/Llama-2-7b-chat-hf` model on a **5k instruction-following dataset** using **QLoRA** (4-bit quantization + LoRA) and TRL’s `SFTTrainer`.  
The goal is to get a lighter, instruction-tuned variant of LLaMA 2 7B that can be trained on a single GPU with limited memory and then published to the Hugging Face Hub. :contentReference[oaicite:1]{index=1}

---

## 💡 What this project does

- Loads the base chat model: `NousResearch/Llama-2-7b-chat-hf`
- Uses a custom instruction dataset from Hugging Face: `Ashok27/guanaco-llama2-5k`
- Applies **QLoRA**:
  - 4-bit quantization using `bitsandbytes`
  - LoRA adapters via `peft.LoraConfig`
- Trains with **TRL’s `SFTTrainer`** for supervised fine-tuning
- Saves the fine-tuned model locally as `Llama-2-7b-chat-finetune`
- Reloads the base model, merges the LoRA weights, and unloads adapters
- Pushes both the final model and tokenizer to the Hugging Face Hub under:
  - `Ashok27/Llama-2-7b-chat-finetune`

---

## 🛠️ Tech stack & libraries

- **Model:** `NousResearch/Llama-2-7b-chat-hf`
- **Dataset:** `Ashok27/guanaco-llama2-5k`
- **Libraries:**
  - `transformers`
  - `datasets`
  - `bitsandbytes`
  - `peft`
  - `trl`
  - `accelerate`
  - `torch`

---

## ⚙️ Key training setup

- 4-bit quantization (`nf4`) with `BitsAndBytesConfig`
- LoRA config:
  - `r = 16`
  - `lora_alpha = 16`
  - `lora_dropout = 0.1`
- Training arguments:
  - `num_train_epochs = 1`
  - `per_device_train_batch_size = 1`
  - `gradient_accumulation_steps = 4`
  - `learning_rate = 2e-4`
  - `weight_decay = 0.001`
  - `lr_scheduler_type = "cosine"`
  - `warmup_ratio = 0.03`
  - `group_by_length = True`
- Uses `DataCollatorForLanguageModeling` with `mlm=False` for causal LM

---

## 📂 Project structure (main script)

- `tuning_llama2_5k.py`  
  - Installs dependencies (for Colab)
  - Loads dataset and model
  - Sets up QLoRA and training configuration
  - Runs `SFTTrainer` on the instruction dataset
  - Saves the trained model
  - Reloads base model and merges LoRA weights
  - Logs in to Hugging Face CLI
  - Pushes model and tokenizer to the Hub

---

## 🚀 How to run

1. Install the required libraries (in Colab or locally):

   ```bash
   pip install -U bitsandbytes
   pip install -q --upgrade accelerate peft bitsandbytes transformers trl datasets
2. Make sure you are logged in to hugging phase
   huggingface-cli login
3. Run the Script
   python tuning_llama2_5k.py
