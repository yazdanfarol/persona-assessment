import pandas as pd
import numpy as np
from pathlib import Path
import json

# Ordinalskala in Zahlen übersetzen
skala = {
    "sehr niedrig": 1, "niedrig": 2, "niedrig-mittel": 2.5,
    "mittel": 3, "mittel-hoch": 3.5, "hoch": 4, "sehr hoch": 5,
    "nur wenn gefragt": 1  # Sonderfall für Direct Doer
}

ziel_profile = {
    "Füllwörter":        {"Gentle Guide": 4, "Easy Expert": 3, "Calm Authority": 2, "Charming Challenger": 4, "Direct Doer": 1},
    "Höflichkeit":       {"Gentle Guide": 4, "Easy Expert": 3.5, "Calm Authority": 4, "Charming Challenger": 3, "Direct Doer": 3},
    "Turnlänge":         {"Gentle Guide": 3, "Easy Expert": 3, "Calm Authority": 2.5, "Charming Challenger": 3, "Direct Doer": 2},
    "Begruendung":       {"Gentle Guide": 3, "Easy Expert": 4, "Calm Authority": 3, "Charming Challenger": 2.5, "Direct Doer": 2},
    "Restating":         {"Gentle Guide": 4, "Easy Expert": 3, "Calm Authority": 2.5, "Charming Challenger": 2, "Direct Doer": 1},
    "Initiative":        {"Gentle Guide": 3, "Easy Expert": 4, "Calm Authority": 4, "Charming Challenger": 4, "Direct Doer": 2},
    "Empathie":          {"Gentle Guide": 5, "Easy Expert": 2.5, "Calm Authority": 2, "Charming Challenger": 3, "Direct Doer": 2},
    "Direktheit":        {"Gentle Guide": 2, "Easy Expert": 3.5, "Calm Authority": 3, "Charming Challenger": 3, "Direct Doer": 5},
    "Humor":             {"Gentle Guide": 1, "Easy Expert": 2, "Calm Authority": 1, "Charming Challenger": 5, "Direct Doer": 1},
    "Slang":             {"Gentle Guide": 1, "Easy Expert": 2, "Calm Authority": 1, "Charming Challenger": 4, "Direct Doer": 2},
    "Enthusiasmus":      {"Gentle Guide": 3, "Easy Expert": 4, "Calm Authority": 2, "Charming Challenger": 4, "Direct Doer": 2},
    "Optionen_anbieten": {"Gentle Guide": 3, "Easy Expert": 4, "Calm Authority": 3, "Charming Challenger": 4, "Direct Doer": 2},
    "Naechster_Schritt":  {"Gentle Guide": 3, "Easy Expert": 4, "Calm Authority": 4, "Charming Challenger": 3.5, "Direct Doer": 1},
}
ziel_df = pd.DataFrame(ziel_profile).T  # Zeilen = Dimensionen, Spalten = Personas

import re
import spacy

nlp = spacy.load("de_dep_news_trf")

# --- Wortlisten (ausbaufähig, aus deinen eigenen Transkripten kalibrieren) ---
FUELLWOERTER = ["also", "genau", "grundsätzlich", "ne", "halt", "sozusagen", "verstehe"]
HOEFLICHKEIT_MARKER = ["bitte", "gerne", "dürfte", "möchten sie", "wäre es möglich", "danke"]
BEGRUENDUNG_MARKER = ["weil", "da ", "damit ", "denn ", "aus diesem grund", "der grund"]
SLANG_MARKER = ["krass", "cool", "mega", "easy", "läuft", "alles klar", "kein ding"]
EMPATHIE_MARKER = ["verstehe", "das tut mir leid", "keine sorge", "kann ich nachvollziehen", "sie müssen sich keine sorgen"]
CTA_MARKER = ["soll ich", "möchten sie, dass ich", "darf ich das gleich", "ich empfehle ihnen zusätzlich"]

def zaehle_marker(text, marker_liste):
    text_l = text.lower()
    return sum(text_l.count(m) for m in marker_liste)

def analysiere_turn(text):
    doc = nlp(text)
    woerter = [t.text for t in doc if not t.is_punct]
    n_woerter = max(len(woerter), 1)

    return {
        "fuellwoerter": zaehle_marker(text, FUELLWOERTER),
        "hoeflichkeit": zaehle_marker(text, HOEFLICHKEIT_MARKER),
        "begruendung": zaehle_marker(text, BEGRUENDUNG_MARKER),
        "slang": zaehle_marker(text, SLANG_MARKER),
        "empathie": zaehle_marker(text, EMPATHIE_MARKER),
        "cta": zaehle_marker(text, CTA_MARKER),
        "fragen": text.count("?"),
        "ausrufe": text.count("!"),
        "wortzahl": n_woerter,
    }

def analysiere_gespraech(agent_turns):
    """agent_turns: Liste von Strings, alle Agenten-Turns EINES Gesprächs"""
    gesamt = {"fuellwoerter":0, "hoeflichkeit":0, "begruendung":0, "slang":0,
              "empathie":0, "cta":0, "fragen":0, "ausrufe":0, "wortzahl":0}
    for turn in agent_turns:
        werte = analysiere_turn(turn)
        for k in gesamt:
            gesamt[k] += werte[k]

    n_turns = max(len(agent_turns), 1)
    return {
        "Füllwörter_pro100w": gesamt["fuellwoerter"] / gesamt["wortzahl"] * 100,
        "Höflichkeit_pro100w": gesamt["hoeflichkeit"] / gesamt["wortzahl"] * 100,
        "Turnlänge_avg": gesamt["wortzahl"] / n_turns,
        "Begründung_pro100w": gesamt["begruendung"] / gesamt["wortzahl"] * 100,
        "Slang_pro100w": gesamt["slang"] / gesamt["wortzahl"] * 100,
        "Empathie_pro100w": gesamt["empathie"] / gesamt["wortzahl"] * 100,
        "Enthusiasmus_pro_turn": gesamt["ausrufe"] / n_turns,
        "Initiative_fragen_pro_turn": gesamt["fragen"] / n_turns,
        "CTA_pro100w": gesamt["cta"] / gesamt["wortzahl"] * 100,
    }

def zaehle_restating(agent_turn, vorheriger_kunden_turn):
    """Grobe Heuristik: Wortüberlappung zwischen Agent-Antwort und Kundenaussage"""
    doc_agent = nlp(agent_turn.lower())
    doc_kunde = nlp(vorheriger_kunden_turn.lower())
    kunden_lemmas = {t.lemma_ for t in doc_kunde if t.pos_ in ("NOUN", "VERB", "PROPN")}
    agent_lemmas = {t.lemma_ for t in doc_agent if t.pos_ in ("NOUN", "VERB", "PROPN")}
    if not kunden_lemmas:
        return 0
    ueberlappung = len(kunden_lemmas & agent_lemmas) / len(kunden_lemmas)
    # Signalwörter für explizites Restating verstärken den Wert
    signal = any(s in agent_turn.lower() for s in ["verstanden", "sie meinen", "sie möchten", "habe ich das richtig"])
    return ueberlappung + (0.5 if signal else 0)

def zaehle_optionen(agent_turn):
    """Zählt aufgezählte Alternativen: 'oder', Kommaaufzählungen mit >=2 Terminen/Produkten"""
    return agent_turn.lower().count(" oder ") + len(re.findall(r",\s*\d", agent_turn))

def extrahiere_agent_turns(daten):
    """Holt aus outputs.messages nur die gesprochenen Agenten-Texte (AIMessage mit content)"""
    

    def extrahiere_messages(agent_turns, messages):
        for msg in messages:
            # AIMessage erkennen
            if msg["id"][-1] != "AIMessage":
                continue
            
            content = msg["kwargs"]["content"]
            
            # content kann String oder Liste sein
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                # Liste von {"type": "text", "text": "..."} Objekten
                text = " ".join(
                    block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
                )
            else:
                text = ""
            
            text = text.strip()
            if text:  # leere Turns (z.B. reine Tool-Calls ohne Text) überspringen
                agent_turns.append(text)
        return  agent_turns
    messages_inputs = daten["daten"]["inputs"]["messages"]
    messages_outputs = daten["daten"]["outputs"]["messages"]
    agent_turns = []
    agent_turns = extrahiere_messages(agent_turns, messages_inputs)
    agent_turns = extrahiere_messages(agent_turns, messages_outputs)
    
    return agent_turns


folder = Path("data")
gespraeche = []

for persona_dir in folder.iterdir():
    if not persona_dir.is_dir():
        continue  # überspringt versteckte Dateien wie .DS_Store
    
    for file in persona_dir.iterdir():
        if file.suffix != ".json":
            continue
        
        with open(file, "r", encoding="utf-8") as f:
            daten = json.load(f)
        
        gespraeche.append({
            "persona": persona_dir.name,       # Ordnername = Persona
            "dateiname": file.name,
            "daten": daten,
        })

print(f"{len(gespraeche)} Gespräche geladen")

# for g in gespraeche:
#     print(g["daten"], "\n")
# test = extrahiere_agent_turns(gespraeche[3])

# for t in test:
#     print("—", t)



ergebnisse = []

for gespraech in gespraeche:
    agent_turns = extrahiere_agent_turns(gespraech)
    werte = analysiere_gespraech(agent_turns)
    
    ergebnisse.append({
        "persona": gespraech["persona"],
        "dateiname": gespraech["dateiname"],
        "n_turns": len(agent_turns),
        **werte
    })

df = pd.DataFrame(ergebnisse)

df_persona = df.groupby("persona").mean(numeric_only=True)

numeric_cols = df_persona.select_dtypes(include="number").columns
df_persona[numeric_cols] = df_persona[numeric_cols].round(2)

df_persona.to_csv("results/2026_09_04.tsv", sep="\t", index=True)
print(df_persona)
