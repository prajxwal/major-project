"""
prediction/abbreviation_expander.py — Context-Aware Abbreviation Expansion.

An intelligent medical assistant that expands user-typed abbreviations
into meaningful, context-aware full sentences. Uses the caretaker's
question and conversation history as context to infer the most likely
intended meaning.

For example:
  Caretaker: "Where did the sound come from?"
  Patient types: "ftk"
  → Suggestions: "from the kitchen", "from the kids", "from the terrace"

Uses Groq LLM (Llama 3.3 70B) for intelligent inference.
"""

import os
import json
import threading
import time
from typing import List, Dict, Callable, Optional
from collections import deque
from dotenv import load_dotenv

load_dotenv()

# Minimum abbreviation length to trigger expansion (single chars are normal typing)
MIN_ABBREV_LENGTH = 2

# Maximum conversation history to keep
MAX_HISTORY = 20


class AbbreviationExpander:
    """
    Context-aware abbreviation expander using Groq LLM.
    
    Maintains a conversation history (caretaker questions + patient responses)
    and uses it as context to expand abbreviations into full sentences.
    """
    
    def __init__(self):
        self._groq_client = None
        self._enabled = False
        self._conversation_history: deque[Dict[str, str]] = deque(maxlen=MAX_HISTORY)
        self._caretaker_context: str = ""  # latest caretaker question/message
        self._last_request_time = 0.0
        self._debounce_ms = 300  # debounce rapid requests
        self._init_groq()
    
    def _init_groq(self):
        """Initialize Groq client."""
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=api_key)
                self._enabled = True
                print("[AbbrevExpander] ✓ Groq LLM abbreviation expander enabled")
            except ImportError:
                print("[AbbrevExpander] ✗ groq package not installed")
            except Exception as e:
                print(f"[AbbrevExpander] ✗ Groq init failed: {e}")
        else:
            print("[AbbrevExpander] ℹ No GROQ_API_KEY, abbreviation expansion disabled")
    
    @property
    def is_enabled(self) -> bool:
        return self._enabled
    
    def set_caretaker_context(self, question: str):
        """
        Set the latest caretaker question/message for context.
        This should be called whenever the caretaker types or speaks.
        """
        self._caretaker_context = question.strip()
        self._conversation_history.append({
            "role": "caretaker",
            "text": question.strip(),
            "time": time.strftime("%H:%M")
        })
    
    def add_patient_message(self, message: str):
        """Record a patient's completed message for conversation history."""
        self._conversation_history.append({
            "role": "patient",
            "text": message.strip(),
            "time": time.strftime("%H:%M")
        })
    
    def is_likely_abbreviation(self, text: str) -> bool:
        """
        Check if the given text looks like an abbreviation that should be expanded.
        
        Heuristics:
        - Length >= MIN_ABBREV_LENGTH
        - Mostly consonants (no/few vowels) → likely abbreviation
        - No spaces (single "word")
        - Not a common English word
        """
        text = text.strip().lower()
        
        if len(text) < MIN_ABBREV_LENGTH:
            return False
        
        # If it has spaces, it's already multi-word — not an abbreviation
        if " " in text:
            return False
        
        # Count vowels vs consonants
        vowels = set("aeiou")
        vowel_count = sum(1 for c in text if c in vowels)
        consonant_count = sum(1 for c in text if c.isalpha() and c not in vowels)
        
        # Very short with no vowels → likely abbreviation (e.g. "ftk", "brb", "pls")
        if len(text) <= 5 and vowel_count == 0 and consonant_count >= 2:
            return True
        
        # Low vowel ratio → likely abbreviation
        if consonant_count > 0 and vowel_count / (vowel_count + consonant_count) < 0.25:
            return True
        
        # Common short abbreviations
        if len(text) <= 4 and consonant_count >= 2:
            # Check against a small set of common real words that look like abbreviations
            real_words = {
                "the", "and", "but", "for", "not", "you", "all", "can",
                "her", "was", "one", "our", "out", "has", "his", "how",
                "its", "may", "new", "now", "old", "see", "way", "who",
                "did", "get", "let", "say", "she", "too", "use", "try",
                "big", "end", "few", "got", "had", "man", "run", "set",
                "top", "put", "red", "yet", "add", "ask", "own", "hot",
                "cut", "dry", "sit", "arm", "leg", "bed", "cry", "yes",
            }
            if text in real_words:
                return False
            return True
        
        return False
    
    def expand(self, abbreviation: str, callback: Callable[[List[str]], None]):
        """
        Expand an abbreviation into full sentence suggestions (async).
        
        Args:
            abbreviation: The short text typed by the patient
            callback: Called with list of expanded sentences when ready
        """
        if not self._enabled or not self._groq_client:
            return
        
        if not self.is_likely_abbreviation(abbreviation):
            return
        
        # Debounce rapid requests
        now = time.time()
        if (now - self._last_request_time) * 1000 < self._debounce_ms:
            return
        self._last_request_time = now
        
        thread = threading.Thread(
            target=self._expand_thread,
            args=(abbreviation, callback),
            daemon=True
        )
        thread.start()
    
    def _expand_thread(self, abbreviation: str, callback: Callable[[List[str]], None]):
        """Background thread for LLM abbreviation expansion."""
        try:
            prompt = self._build_expansion_prompt(abbreviation)
            
            response = self._groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.4,
                max_tokens=400,
                top_p=0.9,
            )
            
            content = response.choices[0].message.content.strip()
            expansions = self._parse_expansions(content)
            
            if expansions:
                callback(expansions)
                
        except Exception as e:
            print(f"[AbbrevExpander] Expansion error: {e}")
    
    def _get_system_prompt(self) -> str:
        """System prompt for the abbreviation expansion LLM."""
        return (
            "You are an intelligent medical communication assistant for ALS patients "
            "who use an eye-gaze typing system called GazeSpeak.\n\n"
            
            "Your job is to expand short abbreviations typed by the patient into "
            "meaningful, context-aware full sentences or phrases.\n\n"
            
            "RULES:\n"
            "1. Use the caretaker's question and conversation history as context.\n"
            "2. Expand abbreviations into 2-4 plausible interpretations.\n"
            "3. Prioritize interpretations that are:\n"
            "   - Relevant to the question asked\n"
            "   - Common in everyday speech\n"
            "   - Contextually logical (location, symptoms, actions, etc.)\n"
            "4. Rank from most likely to least likely.\n"
            "5. Keep responses concise and human-readable.\n"
            "6. Do NOT hallucinate medical facts—stick to reasonable interpretations.\n"
            "7. Each letter in the abbreviation should typically map to the start of a word.\n"
            "   For example: 'ftk' → 'from the kitchen', 'pls' → 'please'\n"
            "8. Some abbreviations are common shorthand: 'brb', 'omw', 'idk', 'ty', 'pls'\n\n"
            
            "RESPONSE FORMAT:\n"
            "Return ONLY a JSON array of objects. Each object must have:\n"
            "- \"expanded_text\": the full phrase/sentence\n"
            "- \"confidence\": number between 0 and 1\n"
            "- \"reasoning\": short explanation\n\n"
            
            "Example response:\n"
            "[\n"
            "  {\"expanded_text\": \"from the kitchen\", \"confidence\": 0.85, "
            "\"reasoning\": \"Each letter matches: f=from, t=the, k=kitchen\"},\n"
            "  {\"expanded_text\": \"from the kids\", \"confidence\": 0.70, "
            "\"reasoning\": \"Relevant if children are present\"}\n"
            "]"
        )
    
    def _build_expansion_prompt(self, abbreviation: str) -> str:
        """Build the expansion prompt with conversation context."""
        parts = []
        
        # Add conversation history
        if self._conversation_history:
            parts.append("CONVERSATION HISTORY:")
            for entry in self._conversation_history:
                role = "Caretaker" if entry["role"] == "caretaker" else "Patient"
                parts.append(f"  [{entry['time']}] {role}: {entry['text']}")
            parts.append("")
        
        # Add current caretaker context
        if self._caretaker_context:
            parts.append(f"CARETAKER'S LATEST QUESTION: \"{self._caretaker_context}\"")
            parts.append("")
        
        # The abbreviation to expand
        parts.append(f"PATIENT TYPED: \"{abbreviation}\"")
        parts.append("")
        parts.append(
            "Expand this abbreviation into 2-4 most likely full phrases/sentences. "
            "Consider each letter as potentially the first letter of a word. "
            "Use the conversation context to rank the most relevant interpretation first."
        )
        
        return "\n".join(parts)
    
    def _parse_expansions(self, content: str) -> List[str]:
        """
        Parse the LLM response into a list of expanded sentences.
        Returns just the expanded_text strings, ordered by confidence.
        """
        try:
            # Try direct JSON parse
            data = json.loads(content)
            if isinstance(data, list):
                # Sort by confidence (descending) and extract text
                sorted_data = sorted(
                    data,
                    key=lambda x: x.get("confidence", 0),
                    reverse=True
                )
                return [
                    item["expanded_text"]
                    for item in sorted_data
                    if "expanded_text" in item
                ]
        except json.JSONDecodeError:
            pass
        
        # Fallback: try to find JSON array in response
        try:
            start = content.index("[")
            end = content.rindex("]") + 1
            data = json.loads(content[start:end])
            if isinstance(data, list):
                sorted_data = sorted(
                    data,
                    key=lambda x: x.get("confidence", 0),
                    reverse=True
                )
                return [
                    item["expanded_text"]
                    for item in sorted_data
                    if "expanded_text" in item
                ]
        except (ValueError, json.JSONDecodeError):
            pass
        
        # Last resort: split lines and clean up
        lines = [line.strip().strip("-•").strip() for line in content.split("\n")]
        return [line for line in lines if line and len(line) > 2 and len(line) < 80]
    
    def get_caretaker_context(self) -> str:
        """Return the current caretaker context."""
        return self._caretaker_context
    
    def clear_context(self):
        """Clear conversation history and context."""
        self._conversation_history.clear()
        self._caretaker_context = ""
