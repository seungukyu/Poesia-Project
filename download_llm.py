import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_type', type=str, default='None')
    parser.add_argument('--hf_model_name', type=str, default='yanolja/EEVE-Korean-Instruct-10.8B-v1.0')
    parser.add_argument('--hf_token', type=str, default='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')
    parser.add_argument('--cache_dir', type=str, default='/home/work/Jungang/hf_llm')
    args = parser.parse_args()
    
    
    ### Llama series
    if hf_model_type == 'llama3.2-3b':
        model_name = 'meta-llama/Llama-3.2-3B-Instruct'
    elif hf_model_type == 'llama3.1-8b':
        model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    elif hf_model_type == 'llama3-8b':
        model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
    elif hf_model_type == 'llama3ko-8b':
        model_name = 'beomi/Llama-3-Open-Ko-8B-Instruct-preview'
        
    ### Qwen series
    elif hf_model_type == 'qwen2.5-7b':
        model_name = 'Qwen/Qwen2.5-7B-Instruct'
    elif hf_model_type == 'qwen2-7b':
        model_name = 'Qwen/Qwen2-7B-Instruct'
        
    ### EXAONE series
    elif hf_model_type == 'EXAONE3.5-7.8b':
        model_name = 'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct'
    elif hf_model_type == 'EXAONE3-7.8b':
        model_name = 'LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct'
    elif hf_model_type == 'None':
        model_name = args.hf_model_name
    
    
    print(f'download model name: {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=args.cache_dir)
