#!/usr/bin/env python3
"""
Simple Gradio chat interface for GPT-OSS-20B model server
Make sure your model server is running on http://localhost:8000 before starting this app
"""

import gradio as gr
import requests
import json
from typing import List, Tuple

# Configuration
MODEL_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-20b"

def chat_with_model(message: str, history: List[Tuple[str, str]], system_prompt: str = "", temperature: float = 0.7, max_tokens: int = 512) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Send a message to the GPT-OSS-20B model and return the response
    """
    try:
        # Build the conversation history for the API
        messages = []
        
        # Add system prompt if provided
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Prepare the request
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        # Make the request
        response = requests.post(
            MODEL_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            assistant_response = result["choices"][0]["message"]["content"]
            
            # Update history
            new_history = history + [(message, assistant_response)]
            return "", new_history
        else:
            error_msg = f"Error {response.status_code}: {response.text}"
            new_history = history + [(message, f"❌ **Error**: {error_msg}")]
            return "", new_history
            
    except requests.exceptions.ConnectionError:
        error_msg = "❌ **Connection Error**: Could not connect to the model server. Make sure it's running on http://localhost:8000"
        new_history = history + [(message, error_msg)]
        return "", new_history
    except requests.exceptions.Timeout:
        error_msg = "❌ **Timeout Error**: The model took too long to respond"
        new_history = history + [(message, error_msg)]
        return "", new_history
    except Exception as e:
        error_msg = f"❌ **Unexpected Error**: {str(e)}"
        new_history = history + [(message, error_msg)]
        return "", new_history

def clear_chat():
    """Clear the chat history"""
    return []

def check_server_status():
    """Check if the model server is running"""
    try:
        response = requests.get("http://localhost:8000/v1/models", timeout=5)
        if response.status_code == 200:
            return "✅ **Server Status**: Online and ready"
        else:
            return f"⚠️ **Server Status**: Server responded with status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ **Server Status**: Offline - Make sure to start your model server first"
    except Exception as e:
        return f"⚠️ **Server Status**: Unknown error - {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="GPT-OSS-20B Chat Interface", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 GPT-OSS-20B Chat Interface")
    gr.Markdown("Chat with your locally hosted GPT-OSS-20B model")
    
    # Server status
    with gr.Row():
        status_text = gr.Markdown(value=check_server_status())
        refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
    
    # Chat interface
    chatbot = gr.Chatbot(
        label="Conversation",
        height=500,
        show_copy_button=True
    )
    
    # Input area
    with gr.Row():
        msg_box = gr.Textbox(
            label="Your message",
            placeholder="Type your message here...",
            scale=4,
            container=False
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)
    
    # Advanced settings
    with gr.Accordion("⚙️ Advanced Settings", open=False):
        system_prompt = gr.Textbox(
            label="System Prompt",
            placeholder="You are a helpful assistant...",
            lines=3
        )
        
        with gr.Row():
            temperature = gr.Slider(
                label="Temperature",
                minimum=0.0,
                maximum=2.0,
                value=0.7,
                step=0.1,
                info="Controls randomness in responses"
            )
            max_tokens = gr.Slider(
                label="Max Tokens",
                minimum=1,
                maximum=2048,
                value=512,
                step=1,
                info="Maximum length of the response"
            )
    
    # Control buttons
    with gr.Row():
        clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
    
    # Event handlers
    def submit_message(message, history, sys_prompt, temp, max_tok):
        if not message.strip():
            return "", history
        return chat_with_model(message, history, sys_prompt, temp, max_tok)
    
    # Button and enter key events
    send_btn.click(
        fn=submit_message,
        inputs=[msg_box, chatbot, system_prompt, temperature, max_tokens],
        outputs=[msg_box, chatbot]
    )
    
    msg_box.submit(
        fn=submit_message,
        inputs=[msg_box, chatbot, system_prompt, temperature, max_tokens],
        outputs=[msg_box, chatbot]
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot]
    )
    
    refresh_btn.click(
        fn=check_server_status,
        outputs=[status_text]
    )

if __name__ == "__main__":
    print("🚀 Starting Gradio chat interface...")
    print("📋 Instructions:")
    print("1. Make sure your GPT-OSS-20B server is running (./your_setup_script.sh)")
    print("2. Open the Gradio interface in your browser")
    print("3. Start chatting!")
    print("\n" + "="*50)
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True if you want a public link
        show_error=True,
        show_api=False
    )