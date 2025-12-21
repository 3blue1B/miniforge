import os
import gradio as gr
import PyPDF2
from openai import OpenAI

# Use an environment variable for security
# You will set this "DEEPSEEK_API_KEY" inside the Render Dashboard
api_key = os.environ.get("DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=api_key, 
    base_url="https://api.deepseek.com/v1"
)
def summarize_and_speak(file_obj):
    if file_obj is None:
        return "Please upload a file.", None

    try:
        # A. Read PDF (Using the Gradio file path)
        reader = PyPDF2.PdfReader(file_obj.name)
        full_summary = ""
        
        # B. Loop through pages (Just like your working script)
        # We limit to first 3 pages for speed, you can remove [:3] for everything
        for page in reader.pages[:3]: 
            page_text = page.extract_text()
            if not page_text: continue
            
            response = client.chat.completions.create(
                model="deepseek-chat", # Using your working model name
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Summarize this: {page_text}"}
                ]
            )
            full_summary += response.choices[0].message.content + "\n\n"

        # C. Generate Audio for the total summary
        audio_path = "summary_audio.mp3"
        audio_response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=full_summary[:4000] # TTS has a 4096 character limit!
        )
        audio_response.stream_to_file(audio_path)
        
        return full_summary, audio_path

    except Exception as e:
        return f"Error: {str(e)}", None

# 4. Interface
interface = gr.Interface(
    fn=summarize_and_speak,
    inputs=gr.File(label="Upload PDF"),
    outputs=[
        gr.Textbox(label="Full Summary", lines=20), 
        gr.Audio(label="Audio Version")
    ]
)

if __name__ == "__main__":
    # Render provides a PORT environment variable, usually 10000
    import os
    port = int(os.environ.get("PORT", 7860)) 
    interface.launch(server_name="0.0.0.0", server_port=port)