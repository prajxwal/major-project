"""
prediction/predictor.py — Word prediction engine.

Provides prefix-based word completion from a local frequency dictionary.
Optionally integrates with Gemini API for smarter context-aware predictions.
"""

import json
import os
from typing import List, Tuple


class WordPredictor:
    """
    Word prediction engine that suggests completions as the user types.
    
    Uses a frequency-ranked dictionary for fast prefix matching.
    """
    
    def __init__(self):
        self._words = []  # list of (word, frequency) sorted by frequency desc
        self._load_dictionary()
    
    def _load_dictionary(self):
        """Load the word frequency dictionary."""
        dict_path = os.path.join(os.path.dirname(__file__), "dictionary.json")
        
        if os.path.exists(dict_path):
            try:
                with open(dict_path, "r") as f:
                    data = json.load(f)
                self._words = [(w["word"], w["freq"]) for w in data]
                # Sort by frequency descending
                self._words.sort(key=lambda x: x[1], reverse=True)
            except (json.JSONDecodeError, KeyError):
                self._words = self._get_fallback_words()
        else:
            self._words = self._get_fallback_words()
            self._save_dictionary()
    
    def predict(self, prefix: str, max_results: int = 5) -> List[str]:
        """
        Get word completions for a given prefix.
        
        Args:
            prefix: The partial word typed so far
            max_results: Maximum number of suggestions to return
            
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
    
    def _save_dictionary(self):
        """Save the dictionary to disk."""
        dict_path = os.path.join(os.path.dirname(__file__), "dictionary.json")
        data = [{"word": w, "freq": f} for w, f in self._words]
        with open(dict_path, "w") as f:
            json.dump(data, f)
    
    def _get_fallback_words(self) -> List[Tuple[str, int]]:
        """Fallback word list if dictionary file is missing."""
        # Top common English words with relative frequency scores
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
            
            # People and relationships
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
            
            # Body parts (medical context)
            ("head", 2500), ("neck", 2400), ("back", 2300), ("chest", 2200),
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
            
            # Questions/Social
            ("name", 2300), ("age", 2290), ("home", 2280), ("work", 2270),
            ("school", 2260), ("phone", 2250), ("television", 2240), ("music", 2230),
            ("book", 2220), ("game", 2210), ("movie", 2200),
            
            # Politeness
            ("excuse", 2100), ("pardon", 2090), ("welcome", 2080),
            ("congratulations", 2070), ("birthday", 2060), ("beautiful", 2050),
            ("wonderful", 2040), ("amazing", 2030), ("great", 2020),
            ("awesome", 2010), ("perfect", 2000),
        ]
        return words
