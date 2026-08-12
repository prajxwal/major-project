"""
prediction/conversation_engine.py — LLM-powered binary conversation engine.

Given a caretaker's question and conversation history, generates exactly
two answer options that the patient can select by looking LEFT or RIGHT.

Each selection narrows the meaning until a complete response is composed.
Uses Groq LLM (Llama 3.3 70B) for intelligent inference.

Example flow:
    Caretaker: "How are you feeling?"
    Round 1: ["I'm okay", "Not great"]          → patient picks "Not great"
    Round 2: ["I have pain", "I feel tired"]     → patient picks "I have pain"
    Round 3: ["In my head", "In my body"]        → patient picks "In my head"
    Final:   🔊 "I'm not great. I have pain in my head."
"""

import os
import re
import json
import threading
import time
from typing import List, Dict, Callable, Optional
from collections import deque
from dotenv import load_dotenv

load_dotenv()

MAX_CONVERSATION_HISTORY = 20
MAX_ROUNDS = 3  # max narrowing rounds before auto-composing


class ConversationEngine:
    """
    LLM backend for binary-choice conversations.

    Maintains:
    - Conversation history (caretaker questions + patient responses)
    - Current round's selection trail (for multi-step narrowing)
    """

    def __init__(self):
        self._groq_client = None
        self._enabled = False
        self._conversation_history: deque[Dict[str, str]] = deque(
            maxlen=MAX_CONVERSATION_HISTORY
        )
        self._current_question: str = ""
        self._selections: List[str] = []  # selections in current round
        self._round = 0
        self._init_groq()

    def _init_groq(self):
        """Initialize Groq client."""
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=api_key)
                self._enabled = True
                print("[ConversationEngine] ✓ Groq LLM conversation engine enabled")
            except ImportError:
                print("[ConversationEngine] ✗ groq package not installed")
            except Exception as e:
                print(f"[ConversationEngine] ✗ Groq init failed: {e}")
        else:
            print("[ConversationEngine] ℹ No GROQ_API_KEY, conversation engine disabled")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def current_question(self) -> str:
        return self._current_question

    @property
    def selections(self) -> List[str]:
        return list(self._selections)

    @property
    def round_number(self) -> int:
        return self._round

    # ─── Public API ─────────────────────────────────────────────

    def start_conversation(self, question: str,
                           callback: Callable[[dict], None]):
        """
        Start a new conversation round with the caretaker's question.
        Generates the first pair of answer options.

        Args:
            question: The caretaker's question/statement
            callback: Called with result dict:
                {
                    "left": str,
                    "right": str,
                    "is_final": bool,
                    "composed_response": str (only if is_final)
                }
        """
        self._current_question = question.strip()
        self._selections = []
        self._round = 0

        # Record caretaker message
        self._conversation_history.append({
            "role": "caretaker",
            "text": question.strip(),
            "time": time.strftime("%H:%M"),
        })

        self._generate_options(callback)

    def select_option(self, chosen_text: str,
                      callback: Callable[[dict], None]):
        """
        Record the patient's selection and generate follow-up options.

        Args:
            chosen_text: The text of the option the patient chose
            callback: Called with next options dict (same format as start_conversation)
        """
        self._selections.append(chosen_text)
        self._round += 1

        if self._round >= MAX_ROUNDS:
            # Force final composition
            self._compose_final(callback)
        else:
            self._generate_options(callback)

    def request_more_options(self, callback: Callable[[dict], None]):
        """Regenerate different options for the current step."""
        self._generate_options(callback, regenerate=True)

    def undo_last_selection(self) -> Optional[str]:
        """Remove the last selection and go back one step.
        Returns the removed selection, or None if nothing to undo."""
        if self._selections:
            removed = self._selections.pop()
            self._round = max(0, self._round - 1)
            return removed
        return None

    def compose_early(self, callback: Callable[[dict], None]):
        """Compose and speak the response from selections so far (early exit)."""
        self._compose_final(callback)

    def record_final_response(self, response: str):
        """Record the patient's final composed response in history."""
        self._conversation_history.append({
            "role": "patient",
            "text": response.strip(),
            "time": time.strftime("%H:%M"),
        })

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Return the full conversation history."""
        return list(self._conversation_history)

    def clear_history(self):
        """Clear all conversation history."""
        self._conversation_history.clear()

    # ─── Private: LLM calls ────────────────────────────────────

    def _generate_options(self, callback: Callable[[dict], None],
                          regenerate: bool = False):
        """Generate answer options in a background thread."""
        if not self._enabled or not self._groq_client:
            # Fallback: provide generic options
            callback(self._fallback_options())
            return

        thread = threading.Thread(
            target=self._generate_options_thread,
            args=(callback, regenerate),
            daemon=True,
        )
        thread.start()

    def _generate_options_thread(self, callback: Callable[[dict], None],
                                 regenerate: bool):
        """Background thread: call Groq LLM to generate 2 options."""
        try:
            print(f"[ConversationEngine] Calling Groq API for: '{self._current_question}'")
            prompt = self._build_options_prompt(regenerate)
            system_prompt = self._get_system_prompt()

            response = self._groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4 if not regenerate else 0.7,
                max_tokens=300,
                top_p=0.9,
                timeout=15.0,
            )

            content = response.choices[0].message.content.strip()
            print(f"[ConversationEngine] LLM raw response: {content[:200]}")
            result = self._parse_options_response(content)

            if result:
                print(f"[ConversationEngine] Parsed OK — "
                      f"left='{result['left']}' right='{result['right']}' "
                      f"is_final={result['is_final']}")
                callback(result)
            else:
                print("[ConversationEngine] Parse failed — using fallback options")
                callback(self._fallback_options())

        except Exception as e:
            print(f"[ConversationEngine] LLM error: {e}")
            callback(self._fallback_options())

    def _compose_final(self, callback: Callable[[dict], None]):
        """Compose the final response from all selections."""
        if not self._enabled or not self._groq_client:
            # Fallback: join selections
            composed = ". ".join(self._selections)
            callback({
                "left": "",
                "right": "",
                "is_final": True,
                "composed_response": composed,
            })
            return

        thread = threading.Thread(
            target=self._compose_final_thread,
            args=(callback,),
            daemon=True,
        )
        thread.start()

    def _compose_final_thread(self, callback: Callable[[dict], None]):
        """Background thread: compose the final spoken response."""
        try:
            prompt = self._build_compose_prompt()

            response = self._groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are composing a natural spoken response for an ALS patient. "
                            "Given the caretaker's question and the patient's selection trail, "
                            "compose a brief, natural-sounding sentence that the patient would say. "
                            "Respond with ONLY a JSON object: "
                            '{"composed_response": "the natural sentence"}'
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=150,
            )

            content = response.choices[0].message.content.strip()
            composed = self._parse_compose_response(content)

            callback({
                "left": "",
                "right": "",
                "is_final": True,
                "composed_response": composed,
            })

        except Exception as e:
            print(f"[ConversationEngine] Compose error: {e}")
            composed = ". ".join(self._selections)
            callback({
                "left": "",
                "right": "",
                "is_final": True,
                "composed_response": composed,
            })

    # ─── Prompt building ───────────────────────────────────────

    def _get_system_prompt(self) -> str:
        return (
            "You are an intelligent communication assistant for ALS patients "
            "using an eye-gaze system called GazeSpeak.\n\n"

            "Your job is to generate exactly TWO answer options that the patient "
            "can choose between by looking LEFT or RIGHT.\n\n"

            "CRITICAL — MINIMIZE PATIENT EFFORT:\n"
            "Each selection costs the patient enormous physical effort (eye strain). "
            "Your #1 goal is to reach a final response in AS FEW ROUNDS AS POSSIBLE.\n"
            "- For simple yes/no questions: finalize IMMEDIATELY after 1 selection.\n"
            "  Example: Q='Do you want water?' → Patient picks 'Yes please' → "
            "  IMMEDIATELY set is_final=true with composed_response='Yes please, I would like some water.'\n"
            "- For open-ended questions: aim for 2 rounds max (e.g., 'How are you?' → "
            "  'Not great' → 'I have pain' → FINALIZE).\n"
            "- NEVER ask unnecessary follow-up details (temperature, container, etc.) "
            "  unless the caretaker's question specifically asks for them.\n"
            "- After 2 selections, STRONGLY prefer is_final=true.\n\n"

            "RULES:\n"
            "1. Generate exactly 2 options — one for LEFT, one for RIGHT.\n"
            "2. Options should be SHORT (2-6 words each).\n"
            "3. Options should cover the most likely contrasting answers.\n"
            "4. Use the caretaker's question and any prior selections as context.\n"
            "5. Prioritize medical, comfort, and daily needs for ALS patients.\n"
            "6. Options should be distinct and non-overlapping.\n"
            "7. Keep language simple and direct.\n\n"

            "RESPONSE FORMAT (JSON only, no other text):\n"
            "{\n"
            '  "left": "short option text",\n'
            '  "right": "short option text",\n'
            '  "is_final": false,\n'
            '  "composed_response": ""\n'
            "}\n\n"

            "If is_final is true, composed_response should be the full natural "
            "sentence the patient wants to say, incorporating all their selections."
        )

    def _build_options_prompt(self, regenerate: bool) -> str:
        parts = []

        # Conversation history
        if self._conversation_history:
            recent = list(self._conversation_history)[-6:]  # last 6 entries
            if recent:
                parts.append("RECENT CONVERSATION:")
                for entry in recent[:-1]:  # skip the current question (already included)
                    role = "Caretaker" if entry["role"] == "caretaker" else "Patient"
                    parts.append(f"  {role}: {entry['text']}")
                parts.append("")

        # Current question
        parts.append(f'CARETAKER\'S QUESTION: "{self._current_question}"')
        parts.append("")

        # Prior selections in this round
        if self._selections:
            parts.append("PATIENT'S SELECTIONS SO FAR:")
            for i, sel in enumerate(self._selections, 1):
                parts.append(f"  Step {i}: \"{sel}\"")
            parts.append("")

            if len(self._selections) >= 2:
                parts.append(
                    "The patient has already made multiple selections. "
                    "Their intent should be clear enough now. "
                    "SET is_final=true AND compose the full response. "
                    "Only generate more options if the intent is truly ambiguous."
                )
            else:
                parts.append(
                    "Generate the next pair of options to narrow down their response. "
                    "If their intent is already clear (especially for simple yes/no questions), "
                    "set is_final=true and compose the full response immediately."
                )
        else:
            parts.append(
                "Generate the first pair of answer options for the patient to choose from."
            )

        if regenerate:
            parts.append(
                "\nIMPORTANT: Generate DIFFERENT options than before — "
                "the patient didn't like the previous ones."
            )

        return "\n".join(parts)

    def _build_compose_prompt(self) -> str:
        parts = [
            f'CARETAKER\'S QUESTION: "{self._current_question}"',
            "",
            "PATIENT'S SELECTIONS:",
        ]
        for i, sel in enumerate(self._selections, 1):
            parts.append(f'  Step {i}: "{sel}"')
        parts.append("")
        parts.append(
            "Compose a brief, natural spoken response that incorporates "
            "all of the patient's selections into one clear sentence."
        )
        return "\n".join(parts)

    # ─── Response parsing ──────────────────────────────────────

    def _parse_options_response(self, content: str) -> Optional[dict]:
        """Parse the LLM JSON response into an options dict."""
        # Strip markdown code fences (e.g. ```json ... ```) that some models add
        content = re.sub(r'```(?:json)?\s*', '', content).strip('`').strip()

        try:
            data = json.loads(content)
            if isinstance(data, dict) and "left" in data and "right" in data:
                return {
                    "left": str(data["left"]).strip(),
                    "right": str(data["right"]).strip(),
                    "is_final": bool(data.get("is_final", False)),
                    "composed_response": str(data.get("composed_response", "")),
                }
        except json.JSONDecodeError:
            pass

        # Fallback: try to find JSON in response
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            data = json.loads(content[start:end])
            if "left" in data and "right" in data:
                return {
                    "left": str(data["left"]).strip(),
                    "right": str(data["right"]).strip(),
                    "is_final": bool(data.get("is_final", False)),
                    "composed_response": str(data.get("composed_response", "")),
                }
        except (ValueError, json.JSONDecodeError):
            pass

        return None

    def _parse_compose_response(self, content: str) -> str:
        """Parse the compose response."""
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "composed_response" in data:
                return data["composed_response"]
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract JSON
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            data = json.loads(content[start:end])
            if "composed_response" in data:
                return data["composed_response"]
        except (ValueError, json.JSONDecodeError):
            pass

        # Last resort: use raw content
        return content.strip().strip('"')

    def _fallback_options(self) -> dict:
        """Generate fallback options when LLM is unavailable."""
        if not self._selections:
            return {
                "left": "Yes",
                "right": "No",
                "is_final": False,
                "composed_response": "",
            }
        elif len(self._selections) == 1:
            return {
                "left": "Tell me more",
                "right": "That's all",
                "is_final": False,
                "composed_response": "",
            }
        else:
            composed = ". ".join(self._selections)
            return {
                "left": "",
                "right": "",
                "is_final": True,
                "composed_response": composed,
            }
