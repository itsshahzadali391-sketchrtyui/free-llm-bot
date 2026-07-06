import gradio as gr
import httpx
import os

ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY", "")

MODELS = {
    "GLM-4 (Best)": "glm-4",
    "GLM-4-Flash (Fast)": "glm-4-flash",
    "GLM-4-Air (Lightweight)": "glm-4-air",
    "GLM-4-Long (Long Text)": "glm-4-long",
    "GLM-4-Plus (Premium)": "glm-4-plus",
}

def chat_with_glm(message, history, model_name, system_prompt, temperature, max_tokens):
    if not ZHIPU_API_KEY:
        return "Error: Zhipu API key not set! Add ZHIPU_API_KEY in Settings > Variables"
    
    model_id = MODELS.get(model_name, "glm-4")
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    for h in history:
        messages.append({"role": "user", "content": h["user"]})
        if h.get("assistant"):
            messages.append({"role": "assistant", "content": h["assistant"]})
    
    messages.append({"role": "user", "content": message})
    
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            result = response.json()
            if "choices" in result:
                return result["choices"][0]["message"]["content"]
            else:
                return str(result)
        else:
            return f"Error {response.status_code}: {response.text[:300]}"
    except Exception as e:
        return f"Error: {str(e)}"

css = ".title{text-align:center;font-size:2em;font-weight:bold}"

with gr.Blocks(css=css, title="GLM Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# GLM Chat\n### Zhipu AI Official API - Free & Unlimited")
    
    chatbot = gr.Chatbot(label="Chat", height=500, type="messages")
    msg = gr.Textbox(label="Message", placeholder="Type here...", lines=2)
    
    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear")
    
    with gr.Accordion("Settings", open=False):
        model_dropdown = gr.Dropdown(
            choices=list(MODELS.keys()),
            value="GLM-4 (Best)",
            label="Model"
        )
        system_prompt = gr.Textbox(
            label="System Prompt",
            value="You are GLM, a helpful AI assistant by Zhipu AI. Be smart, helpful, and accurate.",
            lines=2
        )
        temperature = gr.Slider(0, 2, value=0.7, label="Creativity")
        max_tokens = gr.Slider(100, 4096, value=2048, label="Max Length")
    
    def respond(message, history, model_name, system_prompt, temperature, max_tokens):
        history = history or []
        history.append({"role": "user", "content": message})
        response = chat_with_glm(message, [{"user": m["content"]} for m in history[:-1]], model_name, system_prompt, temperature, max_tokens)
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
