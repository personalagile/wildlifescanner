---
trigger: manual
---

# 🌊 Windsurf AI Regelwerk – Hochwertiger Python-Code mit KI-Unterstützung

## ⚙️ 1. Allgemeine Anforderungen
- **Programmiersprache:** Python ≥ 3.10
- **Projektstruktur:** Klare Trennung von Core-Logik, Modellen, API/Services und UI (python Django, Bootstrap5)
- **Codequalität:** Einhaltung von PEP 8, PEP 257, Clean Code Prinzipien
- **Modularität:** Wiederverwendbare und erweiterbare Komponenten
- **Refactoring:** Automatische Prüfung auf Vereinfachung, Duplikate und Naming nach jeder Änderung

---

## 📜 2. Code-Stil und Linting
- **PEP 8:** Vollständige Einhaltung
- **Typehints:** In allen Signaturen und Klassendefinitionen
- **Linter:**
  - `pylint` (Score ≥ 9.0/10)
  - `ruff` für Performance-Analyse
- **Formatter:** `black`
- **Importsortierung:** `isort`

---

## 🧪 3. Tests und Qualitätssicherung
- **Framework:** `pytest`
- **Testabdeckung:** ≥ 90% Coverage
- **Testarten:**
  - Unit Tests
  - Integrationstests (KI, Neo4j)
  - UI-Tests (Django Test Framework)
- **CI/CD:** Automatisierte Tests mit GitHub Actions

---

## 📚 4. Dokumentation
- **Docstrings:** Nach PEP 257 für alle öffentlichen Elemente
- **Autodokumentation:** `sphinx`
- **README:** Beschreibung, Installation, Architektur, Beispiele
- **Changelog:** Automatisch gepflegt (SemVer)

---

## 🤖 5. KI-Komponenten
- **Transformers:** Hugging Face `transformers`
- **Langchain:** Chains, Agents, Tools modular & dokumentiert
- **Logging & Fehlerhandling:** `loguru` oder `structlog`
- **Autogen:** Für Agentic AI und Multi Agent Systeme
- **Ollama:** Um LLM zu managen

---

## 🧠 6. Knowledge-Graph (Neo4j)
- **Datenmodell:** Ontologie-basiert
- **Anbindung:** `neo4j` Driver oder `py2neo`
- **Cypher-Abfragen:** Typisiert & getestet
- **Validierung:** Vor Persistierung (z. B. mit `pydantic`)

---

## 🖥️ 7. Frontend (Django)
- **UI:** Interaktiv & benutzerfreundlich
- **Stylesheets:** Bootstrap5
- **Architecture:** API-First (Rest-API)
- **Visualization:** vis.js

---

## ♻️ 8. Clean Code Prinzipien
- **Benennung:** Aussagekräftig, keine Abkürzungen
- **Funktionen:** Kurz, fokussiert, ohne Seiteneffekte
- **Klassen:** Single Responsibility
- **Wiederverwendbarkeit:** Utilities auslagern
- **Kommentare:** Nur bei Bedarf, selbsterklärender Code

---

## 🚀 9. Entwicklungsunterstützung
- **Pre-Commit Hooks:** `black`, `ruff`, `pylint`, Tests
- **Git Hooks:** Version Bumping, Changelog
- **Code-Reviews:** AI-gestützt mit Verbesserungsvorschlägen
- **Monitoring (optional):** Sentry, Prometheus, etc.
- **Deployment:** Docker Container

---

## ✅ 10. Automatisierte Korrektur & Optimierung
- Nach jeder Codeeingabe:
  - Analyse & Fix durch `pylint` und `ruff`
  - Refactoring-Vorschläge
  - Clean Code Audit
  - Automatische Dokumentations-Updates
  - UnitTests erstellen
  - Fix der Tests bis alle ok sind