from transformers import LlamaConfig, LlamaForCausalLM

# Initialize Llama model with default configuration
conf = LlamaConfig()
llama = LlamaForCausalLM(conf)
print(llama)

# Calculate and print the total number of trainable parameters
total_params = sum(param.numel() for param in llama.parameters() if param.requires_grad)
print(f'total params: {total_params}')

# Define model dimensions
V, L, H1, H2 = 32000, 32, 4096, 11008  # Llama-7B
# V, L, H1, H2 = 128000, 32, 4096, 14336  # Llama2-8B

# Calculate total parameters by hand and print
total_params_hand = V * H1 + L * (4 * H1**2 + 3 * H1 * H2 + 2 * H1) + H1 + V * H1
print(f'total params by hand cal: {total_params_hand}')