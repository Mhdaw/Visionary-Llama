import streamlit as st
import os
import textwrap
import json
from dotenv import load_dotenv
import librosa
import openai
import pdfplumber
import torch
from transformers import pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from diffusers import StableDiffusion3Pipeline, FluxPipeline
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
from io import StringIO, BytesIO

load_dotenv()
hf_token = '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'

os.environ["HUGGINGFACE_TOKEN"] = hf_token
# Title and description
st.title("Visionary Llama: AI Powerhouse")
st.write("Explore and use various NLP models And audio models and image generation for tasks like summarization, classification, translation, and more.")
st.write("Made by Mhdaw, check out github for more projects")
# Tabs for different NLP tasks
task_tabs = st.tabs(["Llama-academy","Summarization", "Text Classification", "Named Entity Recognition",
                      "Machine Translation", "Text Generation", "Question Answering",
                        "Media Transcription", "Document Summarization", "Visual Question Answering",
                          "Document Question Answering", "Image Generation with prompt enhancement","Image Refiner",
                          "chat with open LLMs"])


class LlamaAcademy:
    def __init__(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "client" not in st.session_state:
            st.session_state.client = None
        if "quiz_active" not in st.session_state:
            st.session_state.quiz_active = False
        if "current_question" not in st.session_state:
            st.session_state.current_question = 0
        if "score" not in st.session_state:
            st.session_state.score = 0
        if "quiz_questions" not in st.session_state:
            st.session_state.quiz_questions = []

    def initialize_client(self, api_key: str, base_url: str):
        st.session_state.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def format_teacher_prompt(self) -> str:
        return """You are a skilled and patient teacher. Your role is to:
1. Explain concepts clearly and thoroughly, starting from the basics
2. Use examples and analogies to make complex topics understandable
3. Break down information into digestible chunks
4. Maintain an encouraging and supportive tone
5. End your explanations with 1-2 review questions about the key points covered

Remember to teach as if you're speaking to a student who is encountering this topic for the first time."""

    def format_quizzer_prompt(self, conversation_history: str) -> str:
        return f"""Based on the following teaching conversation, create 5 multiple-choice questions to test the student's understanding. Format your response as a JSON array with this structure:
[{{
    "question": "Question text",
    "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
    "correct_answer": "A" 
}}]

Teaching conversation:
{conversation_history}"""

    def format_messages(self, query: str, is_quiz: bool = False) -> list:
        messages = [
            {"role": "system", "content": self.format_quizzer_prompt(query) if is_quiz else self.format_teacher_prompt()}
        ]

        if not is_quiz:
            for msg in st.session_state.messages:
                if msg["role"] != "system":
                    messages.append(msg)
            messages.append({"role": "user", "content": query})

        return messages

    def generate_quiz(self, conversation_history: str, model: str):
        messages = self.format_messages(conversation_history, is_quiz=True)
        response = st.session_state.client.chat.completions.create(
            model=model,
            messages=messages
        )
        quiz_content = response.choices[0].message.content
        try:
            st.session_state.quiz_questions = json.loads(quiz_content)
            st.session_state.quiz_active = True
            st.session_state.current_question = 0
            st.session_state.score = 0
        except json.JSONDecodeError:
            st.error("Failed to generate quiz. Please try again.")

    def display_quiz(self):
        if st.session_state.current_question < len(st.session_state.quiz_questions):
            question = st.session_state.quiz_questions[st.session_state.current_question]
            st.write(f"Question {st.session_state.current_question + 1}:")
            st.write(question["question"])

            answer = st.radio("Select your answer:", question["options"], key=f"q_{st.session_state.current_question}")

            if st.button("Submit Answer"):
                selected_letter = answer[0]
                if selected_letter == question["correct_answer"]:
                    st.success("Correct!")
                    st.session_state.score += 1
                else:
                    st.error(f"Incorrect. The correct answer was {question['correct_answer']}")

                st.session_state.current_question += 1
                st.rerun()
        else:
            st.write(f"Quiz completed! Your score: {st.session_state.score}/{len(st.session_state.quiz_questions)}")
            if st.button("Return to Learning"):
                st.session_state.quiz_active = False
                st.rerun()

    def run(self, model=None, input_form=None, processor=None):
        with st.sidebar:
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.session_state.quiz_active = False
        
        api_key = "sk-s2Hpm8x0MkhzV743Ecqzqw"
        base_url = "https://chatapi.akash.network/api/v1"

        if api_key and base_url:
            self.initialize_client(api_key, base_url)

        if st.session_state.quiz_active:
            self.display_quiz()
        else:
            for message in st.session_state.messages:
                if message["role"] != "system":
                    with st.chat_message(message["role"]):
                        st.write(textwrap.fill(message["content"], 100))

            if len(st.session_state.messages) > 0:
                if st.button("Take Quiz"):
                    conversation = "\n".join([msg["content"] for msg in st.session_state.messages
                                               if msg["role"] != "system"])
                    self.generate_quiz(conversation, model)
                    st.rerun()

            if input_form == "text":
                user_input = st.chat_input("Type your message here:", key="text_input")

            if input_form == "audio":
                user_input = st.audio_input("Record your message")

            if user_input:
                if input_form == "audio":
                    # Transcribe voice input
                    prompt = transcribe_speech(user_input, processor, model)
                else:
                    # Use text input
                    prompt = user_input

                # Add user message to session state
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)

                try:
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            messages = self.format_messages(prompt)
                            response = st.session_state.client.chat.completions.create(
                                model=model,
                                messages=messages
                            )
                            assistant_response = response.choices[0].message.content

                            st.write(textwrap.fill(assistant_response, 100))
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": assistant_response
                            })
                except Exception as e:
                    st.error(f"Error: {str(e)}")

#@st.cache_resource
def load_whisper_model(model_name):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    return processor, model

def transcribe_speech(audio_file, processor, model):
    """Transcribe speech from audio file"""
    try:
        audio, sr = librosa.load(audio_file, sr=48000)
        audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        input_features = processor(
            audio_resampled, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features
        
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0] if transcription else ""
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return ""

model_names = ["openai/whisper-tiny", "openai/whisper-tiny.en", "openai/whisper-base", 
                   "openai/whisper-base.en", "openai/whisper-small", "openai/whisper-small.en"]
model_name = model_names[0]
processor, model = load_whisper_model(model_name)

with task_tabs[0]:
    with st.sidebar:
        st.header("Llama academy Configuration")
        col1, col2 = st.columns(2)
        with col1:
            available_models = [
                "Meta-Llama-3-1-8B-Instruct-FP8",
                "Meta-Llama-3-1-405B-Instruct-FP8",
                "Meta-Llama-3-2-3B-Instruct",
                "nvidia-Llama-3-1-Nemotron-70B-Instruct-HF",
            ]
            model = st.selectbox("Select Teacher Model", options=available_models, index=0, key="chat_model")

        with col2:
            input_type = ["text", "audio"]
            input_form = st.selectbox("Conversation Style", input_type, index=0)

    llama_academy = LlamaAcademy()
    llama_academy.run(model=model, input_form=input_form)


# Summarization Tab
with task_tabs[1]:
    st.title("Summarization")
    st.header("Summarization Configuration")
    min_length = st.slider("Minimum Summary Length", min_value=10, max_value=100, value=30, step=5)
    max_length = st.slider("Maximum Summary Length", min_value=50, max_value=500, value=200, step=10)

    input_text = st.text_area("Enter the text to summarize:", "", height=200, key="summarization_text_area")

    @st.cache_resource
    def load_summarizer():
        return pipeline("summarization", model="facebook/bart-large-cnn",device="auto")

    summarizer = load_summarizer()

    if st.button("Summarize", key="summarization"):
        if input_text.strip():
            try:
                summary = summarizer(
                    input_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                st.subheader("Summary")
                st.write(summary[0]['summary_text'])
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter some text to summarize.")

# Text Classification Tab
with task_tabs[2]:
    st.title("Text Classification")
    st.header("Classification Configuration")
    classification_task = st.selectbox("Choose Classification Task", ["Sentiment Analysis", "Topic Classification"])

    input_text = st.text_area("Enter the text for classification:", "", height=200, key="classification_text_area")


    @st.cache_resource
    def load_classifier():
        return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english",device="auto")

    classifier = load_classifier()

    if st.button("Classify", key="classification"):
        if input_text.strip():
            try:
                result = classifier(input_text)
                st.subheader("Classification Result")
                st.json(result)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter some text to classify.")

# Named Entity Recognition Tab
with task_tabs[3]:
    st.header("Named Entity Recognition")
    input_text = st.text_area("Enter the text for NER:", "", height=200, key="ner_text_area")

    @st.cache_resource
    def load_ner_model():
        return pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english",device="auto")

    ner_model = load_ner_model()

    if st.button("Recognize Entities", key="ner"):
        if input_text.strip():
            try:
                entities = ner_model(input_text)
                st.subheader("Recognized Entities")
                st.json(entities)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter some text for NER.")

# Machine Translation Tab
with task_tabs[4]:
    st.header("Machine Translation")
    source_text = st.text_area("Enter the text to translate:", "", height=200, key="translation_text_area")
    source_lang = st.selectbox("Source Language", ["English (en)", "French (fr)"])
    target_lang = st.selectbox("Target Language", ["French (fr)", "English (en)"])

    @st.cache_resource
    def load_translation_model():
        return pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr",device="auto")

    translator = load_translation_model()

    if st.button("Translate", key="translation"):
        if source_text.strip():
            try:
                translation = translator(source_text)
                st.subheader("Translated Text")
                st.write(translation[0]['translation_text'])
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter some text to translate.")

# Text Generation Tab
with task_tabs[5]:
    st.header("Text Generation")
    prompt = st.text_area("Enter a prompt for text generation:", "", height=200, key="generation_text_area")

    max_length = st.slider("Maximum Generated Text Length", min_value=10, max_value=200, value=50, step=10)

    @st.cache_resource
    def load_generator():
        return pipeline("text-generation", model="gpt2",device="auto")

    generator = load_generator()

    if st.button("Generate Text", key="generation"):
        if prompt.strip():
            try:
                generated_text = generator(prompt, max_length=max_length)
                st.subheader("Generated Text")
                st.write(generated_text[0]['generated_text'])
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter a prompt.")

# Question Answering Tab
with task_tabs[6]:
    st.header("Question Answering")
    context = st.text_area("Enter the context:", "", height=200, key="qa_context_text_area")
    question = st.text_area("Enter the question:", "", height=100, key="qa_question_text_area")


    @st.cache_resource
    def load_qa_model():
        return pipeline("question-answering", model="deepset/roberta-base-squad2",device="auto")

    qa_model = load_qa_model()

    if st.button("Get Answer", key="qa"):
        if context.strip() and question.strip():
            try:
                answer = qa_model(question=question, context=context)
                st.subheader("Answer")
                st.write(answer['answer'])
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter both context and question.")

# Media Transcription Tab
with task_tabs[7]:
    st.header("Media Transcription")
    st.write("Transcribe audio files using advanced speech-to-text models.")

    # File type selection
    file_type = "Audio"

    # Sidebar model selection
    st.title("Model Selection")
    model_id = st.selectbox(
        "Choose a model for transcription:",
        [
            "openai/whisper-tiny",
            "openai/whisper-tiny.en",
            "openai/whisper-base",
            "openai/whisper-base.en",
            "openai/whisper-small",
            "openai/whisper-small.en",
            "facebook/wav2vec2-base-960h",
            "facebook/wav2vec2-large-960h",
            "facebook/wav2vec2-large-xlsr-53-english"
        ]
    )

    if file_type == "Audio":
        audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "flac"])

        if audio_file:
            st.write("Audio file uploaded successfully. Processing...")
            audio, sr = librosa.load(audio_file, sr=48000)
            audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            # Model and pipeline setup
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
            )
            model.to(device)

            processor = AutoProcessor.from_pretrained(model_id)

            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=30,
                batch_size=16,  # batch size for inference - set based on your device
                torch_dtype=torch_dtype,
                device='auto',
            )

            result = pipe(audio_resampled)
            st.subheader("Transcription")
            st.write(result['text'])


# Document Summarization Tab
with task_tabs[8]:
    st.header("Document Summarization")
    st.write("Upload a PDF document to generate a summary.")

    # PDF file uploader
    pdf_file = st.file_uploader("Upload a PDF file:", type=["pdf"])

    available_models = [
        "Meta-Llama-3-1-8B-Instruct-FP8",
        "Meta-Llama-3-1-405B-Instruct-FP8",
        "Meta-Llama-3-2-3B-Instruct",
        "nvidia-Llama-3-1-Nemotron-70B-Instruct-HF",
    ]
    model = st.selectbox("Select Model", options=available_models, index=0)

    if pdf_file:
        try:
            # Read the PDF
            with pdfplumber.open(pdf_file) as pdf:
                pdf_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

            st.subheader("Extracted Text")
            st.write(pdf_text[:100] + "..." if len(pdf_text) > 100 else pdf_text)  # Display preview of text
            
            def get_summary(input_text):
                system_prompt = """You are an advanced language model designed to provide detailed and coherent summaries of long texts. Your task is to read the provided text carefully and generate a concise summary that captures the main ideas, key points, and essential details.

Guidelines for Summarization:
1. Comprehension: Understand the overall message and context of the text.
2. Key Points: Identify and highlight the main arguments, themes, or events.
3. Detail Retention: Ensure that important details are included, providing a comprehensive overview.
4. Clarity: Use clear and concise language to convey the summary.
5. Length: Aim for a summary that is approximately 10-20% of the original text length, depending on complexity.
Output Format:
Begin with a brief introduction stating the purpose of the text.
Follow with bullet points or paragraphs that summarize the key ideas.
Conclude with any notable implications or conclusions drawn from the text.
Use correct spacing between words."""
                client = openai.OpenAI(
                    api_key="sk-s2Hpm8x0MkhzV743Ecqzqw",
                    base_url="https://chatapi.akash.network/api/v1"
                )
                response =  client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": input_text},
                    ],
                )
                summary = response.choices[0].message.content
                return summary
        
            if st.button("Summarize Document", key="doc_summarization"):
                try:
                    st.write("Generating summary...")
                    summary = get_summary(pdf_text)
                    # Display summary
                    st.subheader("Summary:")
                    st.write(summary)

                except Exception as e:
                    st.error(f"Error during summarization: {e}")

        except Exception as e:
            st.error(f"Error reading PDF: {e}")

    else:
        st.info("Please upload a PDF document to summarize.")

#Visual Question Answering
with task_tabs[9]:
    st.header("Visual Question Answering")

    if torch.cuda.is_available():
        st.write("Upload an Image and ask your question")

        image_file = st.file_uploader("Upload an Image:",type=["png"])
        question = st.text_area("Ask your question:", "", height=100, key="vqa_question_text_area")


        # Check if an image file has been uploaded
        if image_file is not None:
            image = Image.open(image_file).convert("RGB")

            @st.cache_resource
            def load_vqa_model():
                return pipeline("visual-question-answering",device="auto")
            
            vqa_pipeline = load_vqa_model()
            
            if st.button("Get Answer", key="vqa"):
                if question.strip():
                    try:
                        result = vqa_pipeline(image, question, top_k=1)
                        st.subheader("Answer")
                        st.write(result['answer'])
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("Please enter both image and question.")
        else:
            st.warning("Please upload an image file.")  
    else:
        st.write("GPU is not available, Try other tabs")

#Document Question Answering
with task_tabs[10]:
    st.header("Document Question Answering")

    if torch.cuda.is_available():
        st.write("Upload an Image and ask your question")

        image_file = st.file_uploader("Upload a Document:",type=["png"])
        question = st.text_area("Ask your question:", "", height=100, key="dqa_question_text_area")


        # Check if an image file has been uploaded
        if image_file is not None:
            image = Image.open(image_file).convert("RGB")

            @st.cache_resource
            def load_dqa_model():
                return pipeline("document-question-answering", model="naver-clova-ix/donut-base-finetuned-docvqa",device="auto")
            
            dqa_pipeline = load_dqa_model()
            
            if st.button("Get Answer", key="dqa"):
                if question.strip():
                    try:
                        result = dqa_pipeline(image, question, top_k=1)
                        st.subheader("Answer")
                        st.write(result['answer'])
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("Please enter both image and question.")
        else:
            st.warning("Please upload an image file.")  
    else:
        st.write("GPU is not available, Try other tabs")

#Image Generation with prompt enhancement
# Function to load the appropriate pipeline
@st.cache_resource
def load_imgen_pipeline(model_name):
    if "stable-diffusion" in model_name.lower():
        return StableDiffusion3Pipeline.from_pretrained(model_name)
    elif "flux" in model_name.lower():
        return FluxPipeline.from_pretrained(model_name)
    else:
        raise ValueError("Unsupported model name. Please use Stable Diffusion or Flux model.")

def get_prompt(input_text):
                system_prompt = """You are an expert prompt generator for image creation models such as Stable Diffusion and Flux. Your task is to read the user's input prompt and rewrite it as a more detailed and descriptive version.

1.Enhance Detail: Add relevant details to the prompt, such as specific subjects, environments, artistic styles, colors, lighting, and other attributes that would improve the generated image.
2.Maintain Intent: Ensure the rewritten prompt stays true to the original idea or request from the user while making it more vivid and informative.
3.Adapt for Models: Structure the prompt in a way that aligns with how image generation models interpret text input.
4.Avoid Overloading: Keep the prompt concise enough to avoid overwhelming the model but comprehensive enough to convey clear instructions.
5.When enhancing the prompt, consider including:

1.Artistic style (e.g., photorealistic, surreal, cartoonish, abstract)
2.Scene or setting (e.g., forest at sunrise, futuristic city, underwater world)
3.Subjects or characters (e.g., a smiling cat, a group of astronauts)
4.Lighting and colors (e.g., golden hour, moody lighting, vibrant colors)
5.Mood or theme (e.g., whimsical, mysterious, peaceful)
Output only the enhanced prompt in natural language, ready for use in image generation."""
                client = openai.OpenAI(
                    api_key="sk-s2Hpm8x0MkhzV743Ecqzqw",
                    base_url="https://chatapi.akash.network/api/v1"
                )
                response =  client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": input_text},
                    ],
                )
                prompt = response.choices[0].message.content
                return prompt
# Image Generation with prompt enhancement
with task_tabs[11]:
    st.header("Image Generation with prompt enhancement")
    if torch.cuda.is_available():
        st.write("Generate images using Stable Diffusion or Flux models.")

        # Dropdown for selecting model
        model_name = st.selectbox(
            "Select the model:",
            ["stabilityai/stable-diffusion-3.5-large", "black-forest-labs/FLUX.1-dev"],
        )

        # Text input for prompt
        prompt = st.text_area("Enter your prompt:", key="imgen_prompt_text_area")
        model_prompt = get_prompt(prompt)

        # Generate button
        if st.button("Generate Image"):
            if not model_prompt.strip():
                st.error("Please enter a prompt.")
            else:
                try:
                    # Load the appropriate pipeline
                    st.write("Loading model...")
                    pipe = load_imgen_pipeline(model_name)
                    pipe = pipe.to("cuda")

                    # Generate image
                    st.write("Generating image...")
                    result = pipe(model_prompt).images[0]

                    # Display image
                    st.image(result, caption="Generated Image", use_column_width=True)

                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.write("GPU is not available, Try other tabs")

# Function to load the Stable Diffusion XL Refiner pipeline
@st.cache_resource
def load_refiner_pipeline():
    return StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

# Image Refiner tab
with task_tabs[12]:
    st.header("Image Refinement with Stable Diffusion XL Refiner")
    
    if torch.cuda.is_available():
        st.write("Refine and enhance images using the Stable Diffusion XL Refiner model.")

        # Upload an image
        uploaded_file = st.file_uploader("Upload an image to refine:", type=["jpg", "jpeg", "png"])

        # Text input for prompt
        prompt = st.text_area("Enter your refinement prompt:", key="refiner_prompt_text_area")


        # Refine button
        if st.button("Refine Image"):
            if not uploaded_file or not prompt.strip():
                st.error("Please upload an image and enter a prompt.")
            else:
                try:
                    # Load the refiner pipeline
                    st.write("Loading the refiner model...")
                    pipe = load_refiner_pipeline()
                    pipe = pipe.to("cuda")

                    # Load the image
                    image = Image.open(uploaded_file).convert("RGB")

                    # Generate refined image
                    st.write("Refining image...")
                    refined_image = pipe(prompt=prompt, image=image).images[0]

                    # Display refined image
                    st.image(refined_image, caption="Refined Image", use_column_width=True)

                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.write("GPU is not available. Try other tabs or enable GPU for better performance.")

# Function to load the text generation pipeline
@st.cache_resource
def load_text_generation_pipeline(model_name):
    return pipeline("text-generation", model=model_name,device="auto")

models = [
    # Meta LLaMA Models
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",


    # Qwen Models
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-14B-Instruct",


    # Microsoft Models
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-3.5-MoE-instruct",

    # DeepSeek Models
    "deepseek-ai/DeepSeek-V2-Chat-0628",
    "deepseek-ai/DeepSeek-V2-Chat",
    "deepseek-ai/DeepSeek-V2",
    "deepseek-ai/DeepSeek-V2-Lite",
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "deepseek-ai/DeepSeek-V2.5",
    "deepseek-ai/DeepSeek-Coder-V2-Instruct",
    "deepseek-ai/DeepSeek-Coder-V2-Base",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "deepseek-ai/DeepSeek-Coder-V2-Instruct-0724",

    # Google Models
    "google/gemma-2-2b",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b",
    "google/gemma-2-27b-it",

    # Mistral Models
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "mistralai/Codestral-22B-v0.1"
]
# Chat with Open LLM tab
with task_tabs[13]:
    st.header("Chat with Open LLM")

    if torch.cuda.is_available():
        # Dropdown for selecting model
        model_name = st.selectbox(
            "Select the model:",
            models 
        )

        # Initialize session state for chat history
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Text input for user message
        user_input = st.text_input("Your message:", key="chat_input_text_area")  # Using st.text_input for this case


        # Send button
        if st.button("Send"):
            if not user_input.strip():
                st.error("Please enter a message.")
            else:
                try:
                    # Load the text generation pipeline
                    st.write("Loading model...")
                    text_gen_pipeline = load_text_generation_pipeline(model_name)

                    # Generate response
                    st.write("Generating response...")
                    response = text_gen_pipeline(user_input, max_length=256, num_return_sequences=1)[0]["generated_text"]

                    # Update chat history
                    st.session_state.chat_history.append(("You", user_input))
                    st.session_state.chat_history.append(("LLM", response))

                except Exception as e:
                    st.error(f"Error: {e}")

        # Display chat history
        st.write("### Chat History")
        for speaker, message in st.session_state.chat_history:
            st.write(f"**{speaker}:** {message}")
    else:
        st.write("GPU is not available, try other tabs")
