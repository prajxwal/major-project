"""
prediction/predictor.py — Word prediction engine with Groq LLM integration.

Provides both:
1. Local prefix-based completion (instant, offline)
2. Groq LLM-powered context-aware prediction (smarter, needs internet)

The two are combined: local predictions show instantly, while LLM predictions
arrive async and replace them with smarter suggestions.
"""

import os
import json
import threading
from typing import List, Tuple, Optional, Callable
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class WordPredictor:
    """
    Hybrid word prediction engine.
    
    - Local: instant prefix matching from frequency dictionary
    - LLM: context-aware predictions via Groq API (async)
    """
    
    def __init__(self):
        self._words = []  # list of (word, frequency)
        self._groq_client = None
        self._llm_enabled = False
        self._load_dictionary()
        self._init_groq()
    
    def _init_groq(self):
        """Initialize the Groq client if API key is available."""
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=api_key)
                self._llm_enabled = True
                print("[GazeSpeak] ✓ Groq LLM prediction enabled")
            except ImportError:
                print("[GazeSpeak] ✗ groq package not installed, using local prediction only")
            except Exception as e:
                print(f"[GazeSpeak] ✗ Groq init failed: {e}")
        else:
            print("[GazeSpeak] ℹ No GROQ_API_KEY found, using local prediction only")
    
    def is_llm_enabled(self) -> bool:
        """Check if LLM prediction is available."""
        return self._llm_enabled
    
    def predict(self, prefix: str, max_results: int = 5) -> List[str]:
        """
        Get instant local word completions for a given prefix.
        
        Args:
            prefix: The partial word typed so far
            max_results: Maximum number of suggestions
            
        Returns:
            List of suggested words, ordered by frequency
        """
        if not prefix:
            return []
        
        prefix_lower = prefix.lower()
        results = []
        
        for word, freq in self._words:
            if word.lower().startswith(prefix_lower) and word.lower() != prefix_lower:
                results.append(word)
                if len(results) >= max_results:
                    break
        
        return results
    
    def predict_with_llm(self, full_text: str, current_word: str, 
                         callback: Callable[[List[str]], None],
                         max_results: int = 5):
        """
        Get context-aware predictions from Groq LLM (runs async).
        
        Args:
            full_text: The entire sentence typed so far
            current_word: The word currently being typed (partial)
            callback: Function to call with results when ready
            max_results: Maximum number of suggestions
        """
        if not self._llm_enabled or not self._groq_client:
            return
        
        # Run in background thread to avoid blocking UI
        thread = threading.Thread(
            target=self._llm_predict_thread,
            args=(full_text, current_word, callback, max_results),
            daemon=True
        )
        thread.start()
    
    def _llm_predict_thread(self, full_text: str, current_word: str,
                             callback: Callable[[List[str]], None],
                             max_results: int):
        """Background thread for LLM prediction."""
        try:
            prompt = self._build_prompt(full_text, current_word, max_results)
            
            response = self._groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a word prediction assistant for an ALS patient "
                            "using an eye-gaze typing system. Your job is to predict "
                            "the most likely next words or word completions to minimize "
                            "the number of characters the patient needs to type. "
                            "Respond with ONLY a JSON array of word suggestions, "
                            "nothing else. Example: [\"hello\", \"help\", \"here\"]"
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=100,
                top_p=0.9,
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            words = self._parse_llm_response(content)
            
            if words:
                callback(words[:max_results])
                
        except Exception as e:
            print(f"[GazeSpeak] LLM prediction error: {e}")
    
    def _build_prompt(self, full_text: str, current_word: str, max_results: int) -> str:
        """Build the prediction prompt for the LLM."""
        if current_word:
            return (
                f"The patient has typed: \"{full_text}\"\n"
                f"They are currently typing the word: \"{current_word}\"\n"
                f"Suggest {max_results} most likely completions or next words. "
                f"If the current word is partial, suggest completions for it. "
                f"Consider the context of ALS patient communication (medical needs, "
                f"comfort, social interaction). Return ONLY a JSON array."
            )
        else:
            return (
                f"The patient has typed: \"{full_text}\"\n"
                f"Suggest {max_results} most likely next words they might want to type. "
                f"Consider the context of ALS patient communication (medical needs, "
                f"comfort, social interaction). Return ONLY a JSON array."
            )
    
    def _parse_llm_response(self, content: str) -> List[str]:
        """Parse the LLM response into a list of words."""
        try:
            # Try direct JSON parse
            words = json.loads(content)
            if isinstance(words, list):
                return [str(w).strip() for w in words if w]
        except json.JSONDecodeError:
            pass
        
        # Fallback: try to find JSON array in response
        try:
            start = content.index("[")
            end = content.index("]") + 1
            words = json.loads(content[start:end])
            if isinstance(words, list):
                return [str(w).strip() for w in words if w]
        except (ValueError, json.JSONDecodeError):
            pass
        
        # Last resort: split by commas or newlines
        words = [w.strip().strip('"\'') for w in content.replace('\n', ',').split(',')]
        return [w for w in words if w and len(w) < 30]
    
    def predict_next_word(self, full_text: str, callback: Callable[[List[str]], None],
                          max_results: int = 5):
        """
        Predict the next word after a completed word/sentence.
        
        This is called when the user adds a space, to suggest what comes next.
        """
        if not self._llm_enabled:
            # Fallback: suggest common follow-up words
            common = ["I", "the", "a", "to", "is", "and", "my", "please", "can", "need"]
            callback(common[:max_results])
            return
        
        self.predict_with_llm(full_text, "", callback, max_results)
    
    def _load_dictionary(self):
        """Load the word frequency dictionary."""
        dict_path = os.path.join(os.path.dirname(__file__), "dictionary.json")
        
        if os.path.exists(dict_path):
            try:
                with open(dict_path, "r") as f:
                    data = json.load(f)
                self._words = [(w["word"], w["freq"]) for w in data]
                self._words.sort(key=lambda x: x[1], reverse=True)
            except (json.JSONDecodeError, KeyError):
                self._words = self._get_fallback_words()
        else:
            self._words = self._get_fallback_words()
            self._save_dictionary()
    
    def _save_dictionary(self):
        """Save the dictionary to disk."""
        dict_path = os.path.join(os.path.dirname(__file__), "dictionary.json")
        data = [{"word": w, "freq": f} for w, f in self._words]
        with open(dict_path, "w") as f:
            json.dump(data, f)
    
    def _get_fallback_words(self) -> List[Tuple[str, int]]:
        """Fallback word list if dictionary file is missing."""
        words = [
            # Basic communication
            ("yes", 9900), ("no", 9800), ("help", 9700), ("please", 9600),
            ("thank", 9500), ("thanks", 9490), ("sorry", 9400), ("hello", 9300),
            ("goodbye", 9200), ("okay", 9100), ("ok", 9090),
            
            # Essential needs
            ("water", 9000), ("food", 8900), ("pain", 8800), ("cold", 8700),
            ("hot", 8600), ("tired", 8500), ("sleep", 8400), ("rest", 8300),
            ("bathroom", 8200), ("medicine", 8100), ("doctor", 8000),
            ("nurse", 7900), ("hospital", 7800), ("comfortable", 7700),
            ("uncomfortable", 7690), ("hungry", 7600), ("thirsty", 7500),
            
            # People
            ("love", 7400), ("family", 7300), ("friend", 7200), ("mom", 7100),
            ("dad", 7000), ("wife", 6900), ("husband", 6800), ("son", 6700),
            ("daughter", 6600), ("child", 6500), ("children", 6490), ("baby", 6400),
            
            # Feelings
            ("happy", 6300), ("sad", 6200), ("scared", 6100), ("angry", 6000),
            ("worried", 5900), ("better", 5800), ("worse", 5700), ("good", 5600),
            ("bad", 5500), ("fine", 5400),
            
            # Actions
            ("want", 5300), ("need", 5200), ("have", 5100), ("like", 5000),
            ("go", 4900), ("come", 4800), ("see", 4700), ("know", 4600),
            ("think", 4500), ("feel", 4400), ("tell", 4300), ("ask", 4200),
            ("call", 4100), ("move", 4000), ("turn", 3900), ("open", 3800),
            ("close", 3700), ("stop", 3600), ("start", 3500), ("wait", 3400),
            ("read", 3300), ("watch", 3200), ("listen", 3100), ("eat", 3000),
            ("drink", 2900), ("sit", 2800), ("stand", 2700), ("walk", 2600),
            
            # Common words
            ("the", 10000), ("is", 9950), ("and", 9940), ("to", 9930),
            ("in", 9920), ("it", 9910), ("that", 9895), ("was", 9890),
            ("for", 9885), ("on", 9880), ("are", 9875), ("with", 9870),
            ("this", 9865), ("but", 9860), ("not", 9855), ("you", 9850),
            ("all", 9845), ("can", 9840), ("her", 9835), ("him", 9830),
            ("one", 9825), ("our", 9820), ("out", 9815), ("day", 9810),
            ("get", 9805), ("has", 9800), ("my", 9795), ("your", 9790),
            ("what", 9785), ("when", 9780), ("who", 9775), ("how", 9770),
            ("where", 9765), ("why", 9760), ("would", 9755), ("could", 9750),
            ("should", 9745), ("will", 9740), ("just", 9735), ("more", 9730),
            ("some", 9725), ("time", 9720), ("very", 9715), ("about", 9710),
            ("which", 9705), ("they", 9700), ("been", 9695), ("from", 9690),
            ("make", 9685), ("only", 9680), ("than", 9675), ("other", 9670),
            ("into", 9665), ("back", 9660), ("much", 9655), ("also", 9650),
            
            # Body parts
            ("head", 2500), ("neck", 2400), ("chest", 2200),
            ("arm", 2100), ("hand", 2000), ("leg", 1900), ("foot", 1800),
            ("stomach", 1700), ("throat", 1600), ("eyes", 1500), ("mouth", 1400),
            
            # Position/comfort
            ("up", 2450), ("down", 2440), ("left", 2430), ("right", 2420),
            ("position", 2410), ("pillow", 2400), ("blanket", 2390),
            ("light", 2380), ("dark", 2370), ("loud", 2360), ("quiet", 2350),
            ("adjust", 2340), ("change", 2330),
            
            # Time
            ("now", 2500), ("later", 2490), ("today", 2480), ("tomorrow", 2470),
            ("morning", 2460), ("night", 2450), ("soon", 2440),
        ]
        return words
