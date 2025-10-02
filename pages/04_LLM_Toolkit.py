# app_llm_toolkit.py
import streamlit as st

st.set_page_config(page_title="LLM Theory Toolkit", page_icon="🤖", layout="wide")

st.title("🤖 LLM Theory Toolkit")
st.markdown("""
- LLM (Large Language Model) is a AI model that has been trained using a vast amount of text in order to understand and generate text. 
- Typical use cases include: text extraction, summarization, Q&A, translation, classification and chatbots. 
- Their architecture is mostly based in **Transformers** with billions of parámeters. 
""")
st.divider()

tabs = st.tabs([
    "2) Transformer Architecture",
    "3) Pipelines (HF)",
    "4) AutoClasses",
    "5) LLM Lifecycle & Tokenization",
    "6) Fine-tuning (theory + Trainer)",
    "7) Prompting & Decoding",
    "8) Evaluation & Guardrails",
    "9) Deploy: Hugging Face",
    "10) Deploy: Ollama (local)"
])

# 2) Transformer Architecture
with tabs[0]:
    st.subheader("Transformer Architecture")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Transformers** process sequences in parallel using **self-attention**:
- Capture long-range dependencies without recurrence.
- Scale to large contexts (limited per model).
        """)
        st.markdown("**Main Families:**")
        st.markdown("""
- **Encoder-only (BERT):** understanding, classification, *extractive QA*.
- **Decoder-only (GPT):** free-form generation, *generative QA*.
- **Encoder-Decoder (T5/BART):** input→output transformations (translation, *summarization*).
        """)
    with c2:
        st.code("""
# Check the architecture of a model in Transformers
from transformers import AutoModel

m = AutoModel.from_pretrained("bert-base-uncased")
print(m.config.architectures)  # e.g., ['BertModel']
        """, language="python")
    st.warning("Rule of thumb: Encoder → understanding, Decoder → generation, Enc-Dec → sequence-to-sequence transformations.")
    st.divider()


# 3) Pipelines (HF)
with tabs[1]:
    st.subheader("Hugging Face Pipelines: quick inference")
    st.markdown("""
**Pipelines** simplify inference with automatic model/tokenizer selection. Useful for prototypes.
    """)
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("**Summarization**")
        st.code("""
from transformers import pipeline

summarizer = pipeline(task="summarization", model="facebook/bart-large-cnn")
text = "Your long text here..."
summary = summarizer(text, max_length=150, min_length=60, truncation=True)
print(summary[0]["summary_text"])
        """, language="python")
    with colB:
        st.markdown("**Text Generation**")
        st.code("""
from transformers import pipeline

generator = pipeline(task="text-generation", model="distilgpt2")
prompt = "Write a haiku about oceans:"
out = generator(
    prompt,
    max_length=80,
    pad_token_id=generator.tokenizer.eos_token_id,
    do_sample=True, top_p=0.9, temperature=0.8
)
print(out[0]["generated_text"])
        """, language="python")
    with colC:
        st.markdown("**Translation EN→ES**")
        st.code("""
from transformers import pipeline

translator = pipeline(
    task="translation_en_to_es",
    model="Helsinki-NLP/opus-mt-en-es"
)
out = translator("This is a test.", clean_up_tokenization_spaces=True)
print(out[0]["translation_text"])
        """, language="python")
    st.divider()
    st.markdown("**Useful Parameters**")
    st.code("""
# Quick tips:
# - truncation=True if input exceeds the context window
# - pad_token_id = tokenizer.eos_token_id to fill with EOS
# - do_sample + top_p/top_k + temperature for diversity
    """)
    st.info("Limitation: less fine control. For customization, use AutoClasses.")
    st.divider()

# 4) AutoClasses
with tabs[2]:
    st.subheader("AutoClasses: more control than Pipelines")
    st.markdown("""
**AutoTokenizer/AutoModel** allow customization of *heads*, *logits*, *padding*, *masking*, etc.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.code("""
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

enc = tok(["a good movie", "a bad movie"],
          padding=True, truncation=True, return_tensors="pt")
logits = model(**enc).logits
        """, language="python")
    with col2:
        st.code("""
# AutoModelForCausalLM for text generation
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("distilgpt2")
m = AutoModelForCausalLM.from_pretrained("distilgpt2")
inp = tok("Hello, my name is", return_tensors="pt")
ids = m.generate(**inp, max_length=60)
print(tok.decode(ids[0], skip_special_tokens=True))
        """, language="python")
    st.divider()

# 5) LLM Lifecycle & Tokenization
with tabs[3]:
    st.subheader("LLM Lifecycle & Tokenization")
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown("""
**Lifecycle:**
1. **Pre-training:** large-scale self-supervised general learning.
2. **Fine-tuning:** domain/task specialization (legal, customer support, healthcare).
3. *(Optional)* **Instruction tuning / RLHF:** alignment to human instructions.

**Tokenization (subword):**
- BPE / WordPiece / Unigram: split words into frequent sub-parts.
- Process in batches for better performance.
- Dataset *sharding* improves training throughput.
        """)
    with c2:
        st.code("""
# Example of batched tokenization with datasets
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("imdb")
tok = AutoTokenizer.from_pretrained("bert-base-uncased")

def tok_fn(ex):
    return tok(ex["text"], truncation=True, padding="max_length", max_length=256)

ds_tok = ds.map(tok_fn, batched=True)
        """, language="python")
    st.divider()

# 6) Fine-tuning (theory + Trainer)
with tabs[4]:
    st.subheader("Fine-tuning: approaches and Trainer")
    st.markdown("""
**Approaches:**
- **Full fine-tuning:** update all weights (expensive).
- **Partial fine-tuning:** freeze layers and adjust only a few (faster).
- **PEFT/LoRA/QLoRA:** inject low-rank adapters (optimal cost/performance).
- **Transfer learning:** adapt a pre-trained model to a related task.

**Key hyperparameters:**
- `learning_rate`, `num_train_epochs`, `per_device_train_batch_size`, `gradient_accumulation_steps`.
- `evaluation_strategy` (`'steps'| 'epoch' | 'no'`), `save_strategy`, `logging_steps`.
    """)
    st.code("""
# Trainer skeleton (binary classification)
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from datasets import load_dataset
import numpy as np
from evaluate import load as load_metric

ds = load_dataset("imdb")
tok = AutoTokenizer.from_pretrained("bert-base-uncased")

def tok_fn(ex):
    return tok(ex["text"], truncation=True, padding="max_length", max_length=256)

ds = ds.map(tok_fn, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

args = TrainingArguments(
    output_dir="out",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

acc = load_metric("accuracy")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return acc.compute(predictions=preds, references=p.label_ids)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    compute_metrics=compute_metrics
)

trainer.train()
    """, language="python")
    st.info("N-shot prompting (zero/one/few-shot) = **without** training, only examples in the prompt.")
    st.divider()

# 7) Prompting & Decoding
with tabs[5]:
    st.subheader("Prompting and decoding control")
    colL, colR = st.columns(2)
    with colL:
        st.markdown("""
**Prompting principles:**
- Be specific: define role/style/output format.
- Bound the length: numbered steps, bullets, JSON if needed.
- Provide examples (*few-shot*) and counterexamples if useful.

**Decoding:**
- **Greedy**: deterministic, less creative.
- **Beam search**: explores best sequences (may overfit).
- **Nucleus/Top-p** and **Top-k**: controlled diversity.
- **Temperature**: smooths or sharpens probability distribution.
        """)
    with colR:
        st.code("""
# Example of decoding with causal generation
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("distilgpt2")
m = AutoModelForCausalLM.from_pretrained("distilgpt2")

inp = tok("Give 3 bullet points about oceans:", return_tensors="pt")
ids = m.generate(
    **inp,
    max_length=120,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    eos_token_id=tok.eos_token_id
)
print(tok.decode(ids[0], skip_special_tokens=True))
        """, language="python")
    st.warning("If the prompt is vague, the output will be suboptimal. Iterate on your prompt.")
    st.divider()

# 8) Evaluation & Guardrails
with tabs[6]:
    st.subheader("Evaluation and Guardrails")
    st.markdown("""
**Metrics per task:**
- **QA**: Exact Match, F1.
- **Summarization**: ROUGE.
- **Translation**: BLEU/COMET.
- **Classification**: accuracy/F1/ROC-AUC.

**Guardrails (safety and reliability):**
- Filtering of *prompt injections* and PII.
- Toxicity checks, jailbreak attempts.
- Truthfulness: retrieval or external verification.
- Monitoring: *latency*, *drift*, hallucination rate.
    """)
    st.code("""
# Simple evaluation skeleton (classification)
# (Assumes compute_metrics in Trainer using accuracy/F1)
# For QA/Summarization, use 'evaluate' with specific metrics (exact_match, rouge).
    """)
    st.divider()

# 9) Deploy: Hugging Face
with tabs[7]:
    st.subheader("Deployment in the Hugging Face ecosystem")
    st.markdown("""
**Common options:**
- **Inference API / Endpoints**: serve models as REST.
- **Spaces**: apps with Gradio/Streamlit for demos.
- **Model cards**: recommended model documentation.

**Typical flow:**
1) Upload model and *tokenizer* to the Hub.  
2) Create a Space (Streamlit/Gradio) for demo.  
3) (Production) Configure an **Endpoint** with autoscaling.
    """)
    st.code("""
# Minimal REST client example (pseudocode)
import requests

URL = "https://api-inference.huggingface.co/models/ORG/MODEL"
headers = {"Authorization": "Bearer hf_xxx"}
payload = {"inputs": "Explain transformers in one sentence."}

resp = requests.post(URL, headers=headers, json=payload, timeout=60)
print(resp.json())
    """, language="python")
    st.info("Advantages: quick way to share prototypes. Consider costs and latency depending on model size.")
    st.divider()

# 10) Deploy: Ollama (local)
with tabs[8]:
    st.subheader("Ollama: running models locally")
    st.markdown("""
**Ollama** lets you run LLMs on your machine (privacy and cost control).
- Common models: `llama3`, `mistral`, `qwen`.
- Great for prototyping and offline development.

**Basic commands:**
- `ollama pull llama3`
- `ollama run llama3`
    """)
    colx, coly = st.columns(2)
    with colx:
        st.code("""
# Python + Ollama REST
import requests

resp = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3", "prompt": "Write a haiku about oceans."}
)
print(resp.json()["response"])
        """, language="python")
    with coly:
        st.code("""
# LangChain + Ollama (basic example)
from langchain_community.llms import Ollama

llm = Ollama(model="llama3")
print(llm.invoke("List 3 pros of local inference."))
        """, language="python")
    st.warning("Limitations: context length and speed depend on your hardware. Adjust model size to your resources.")
    st.divider()