# Fine_Tuning_LlaMA2_using_Guanaco_Dataset
Fine-tuning LLaMA-2 7B using a 5k instruction dataset with QLoRA (4-bit quantization + LoRA) to reduce GPU usage and enable training on a single machine. The model is trained with TRL’s SFTTrainer, merged with base weights, and uploaded to the Hugging Face Hub for easy use and deployment.
