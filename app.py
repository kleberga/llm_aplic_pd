#################################################################
#
# Script para criar um dashboard de auxíli de aprendizagem
#
#################################################################

import streamlit as st
from agente_cursos import Agent
from transformers import pipeline

st.set_page_config(layout="wide")
st.title("Bem vindo ao aplicativo de auxílio no aprendizado!")
st.write("O objetivo deste aplicativo é fornecer sugestões de sites para que você aprenda mais sobre o assunto desejado.")

agent = Agent("AIzaSyCbWuENZJLpPx82w-Ju9Qd1a4xHqyi45lo")

col1, col2 = st.columns(2)

with col1:
    request = st.text_area("Descreva o assunto que você deseja aprender mais, em português.")
    button = st.button("Obter as sugestões")
    box = st.container()
    with box:
        container = st.empty()
        container.write("Sugestões de Sites:")

# modelo para detectar prompt injection
detector = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"  # Modelo NLI para detecção de intenção
)

# função para detectar prompt injection
def detect_prompt_injection(text):
    candidate_labels = ["prompt injection", "normal query"]
    result = detector(text, candidate_labels)
    print(result)
    label = result['labels'][0]  
    return label == "prompt injection"

# função para definir a ação em caso de prompt injection positivo ou negativo
def safe_agent(question):
    is_injection = detect_prompt_injection(question)
    if is_injection:
        return {"learning_sites": "🚫 Prompt Injection Detectado. Operação bloqueada."}
    else:
        return agent.get_tips(question)

if button:
    if request:
        # sites = agent.get_tips(request)
        sites = safe_agent(request)
        container.write(sites["learning_sites"])

