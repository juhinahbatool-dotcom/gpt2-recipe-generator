import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import gdown
import zipfile

# -------------------------------
# CONFIGURATION
# -------------------------------

FILE_ID = "1rMqY_RfLSY_Y-OnQ9wM4fW25kPRVZs_C"
DRIVE_FILE_URL = f"https://drive.google.com/uc?id={FILE_ID}"
MODEL_ZIP_PATH = "model.zip"
MODEL_DIR = "fine_tuned_gpt2_recipes"

# -------------------------------
# DOWNLOAD AND EXTRACT MODEL
# -------------------------------
@st.cache_resource
def load_model():
    # Check if already extracted
    if not os.path.exists(MODEL_DIR):
        st.write("Downloading model from Google Drive...")
        gdown.download(DRIVE_FILE_URL, MODEL_ZIP_PATH, quiet=False)

        st.write("Extracting model files...")
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)

    # Load model and tokenizer
    st.write("Loading GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("AI Recipe Generator (GPT-2 Fine-Tuned)")
st.markdown("Enter a dish name or ingredients to generate a recipe.")

input_text = st.text_area("🧂 Enter dish name or ingredients:", height=100, placeholder="e.g., chicken, garlic, butter, lemon")

max_length = st.slider("Recipe length", 50, 400, 150)
generate_btn = st.button("Generate Recipe")

# -------------------------------
# GENERATION LOGIC
# -------------------------------
if generate_btn:
    if input_text.strip() == "":
        st.warning("Please enter some ingredients or dish name.")
    else:
        prompt = f"Recipe for {input_text}:\nIngredients:\n"
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        with st.spinner("Generating recipe... please wait..."):
            output = model.generate(
                input_ids,
                max_length=max_length,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            # Optional: clean up or truncate output here

        st.subheader("🍽️ Generated Recipe:")
        st.write(generated_text)


# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Built with using GPT-2 Fine-Tuned Model | Uploaded from Google Drive")
