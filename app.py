import gradio as gr
import httpx
import json
import os

HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")

MODELS = {
    "GLM-5.2 (Best)": "zhipu/GLM-4-9B-Chat-1M",
    "Qwen2.5-72B (ChatGPT Level)": "Qwen/Qwen2.5-72B-Instruct",
    "DeepSeek-V3": "deepseek-ai/DeepSeek-V3-0324",
    "Llama-3.1-70B": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "Mistral-Large": "mistralai/Mistral-Large-Instruct-2407",
    "Qwen2.5-32B (Fast)": "Qwen/Qwen2.5-32B-Instruct",
    "Llama-3.1-8B (Ultra Fast)": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}

def chat_with_llm(message, history, model_name, system_prompt, temperature, max_tokens):
    if not HF_API_TOKEN:
        return "Error: HuggingFace API token not set! Go to Settings > Repository Secrets > Add HF_API_TOKEN"
    
    model_id = MODELS.get(model_name, "zhipu/GLM-4-9B-Chat-1M")
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    for h in history:
        messages.append({"role": "user", "content": h["user"]})
        if h.get("assistant"):
            messages.append({"role": "assistant", "content": h["assistant"]})
    
    messages.append({"role": "user", "content": message})
    
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result:
                return result["choices"][0]["message"]["content"]
            elif "generated_text" in result:
                return result["generated_text"]
            else:
                return json.dumps(result)
        else:
            error_detail = response.text[:500] if response.text else "No details"
            return f"API Error {response.status_code}: {error_detail}"
    except Exception as e:
        return f"Error: {str(e)}"

css = """
.container { max-width: 900px; margin: auto; }
"""

with gr.Blocks(css=css, title="Free LLM Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Free LLM Chat
    ### GLM-5.2 | Qwen2.5-72B | DeepSeek-V3 | Llama-3.1-70B
    **FREE + Unlimited + ChatGPT Level Quality**
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat", height=500, type="messages")
            msg = gr.Textbox(label="Message", placeholder="Type your message...", lines=2)
            
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")
        
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="GLM-5.2 (Best)",
                label="Model"
            )
            system_prompt = gr.Textbox(
                label="System Prompt",
                value="You are a helpful, accurate, and intelligent AI assistant. Answer in the same language as the user. You are GLM-5.2, powered by Zhipu AI.",
                lines=3
            )
            temperature = gr.Slider(0, 2, value=0.7, label="Temperature")
            max_tokens = gr.Slider(100, 4096, value=2048, label="Max Tokens")
    
    gr.Markdown("""
    ---
    ### Models:
    | Model | Quality | Speed |
    |-------|---------|-------|
    | **GLM-5.2** | Best Chinese LLM | Medium |
    | Qwen2.5-72B | ChatGPT Level | Medium |
    | DeepSeek-V3 | Best Open Source | Slow |
    | Llama-3.1-70B | Excellent | Medium |
    | Mistral-Large | Excellent | Medium |
    | Qwen2.5-32B | Very Good | Fast |
    | Llama-3.1-8B | Good | Ultra Fast |
    """)
    
    def respond(message, history, model_name, system_prompt, temperature, max_tokens):
        history = history or []
        history.append({"role": "user", "content": message})
        
        response = chat_with_llm(message, [{"user": m["content"]} for m in history[:-1]], model_name, system_prompt, temperature, max_tokens)
        
        history.append({"role": "assistant", "content": response})
        return "", history, ""
    
    def clear_chat():
        return [], ""
    
    msg.submit(respond, [msg, chatbot, model_dropdown, system_prompt, temperature, max_tokens], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot, model_dropdown, system_prompt, temperature, max_tokens], [msg, chatbot])
    clear_btn.click(clear_chat, outputs=[chatbot, msg])

app = demo

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
