import joblib
import re


class Correction:
    def __init__(self):
        self.freq = joblib.load('data/vocab_freq.gz')
        self.vocab = list(self.freq.keys())
        m = sum(self.freq.values())
        self.probs = {k: v/m for k, v in self.freq.items()}
        self.alphabet = 'آ.ا.ب.پ.ت.ث.ج.چ.ح.خ.د.ذ.ر.ز.ژ.س.ش.ص.ض.ط.ظ.ع.غ.ف.ق.ک.گ.ل.م.ن.و.ه.ی'.split('.')

    @staticmethod
    def _splits(word):
        return [(word[:i], word[i:]) for i in range(len(word)+1)]

    def _deletes(self, word):
        return [L + R[1:] for L, R in self._splits(word) if R]

    def _switches(self, word):
        return [L[:-1] + R[0] + L[-1] + R[1:] for L, R in self._splits(word) if L and R]

    def _inserts(self, word):
        return [L + c + R for L, R in self._splits(word) for c in self.alphabet]

    def _replaces(self, word):
        return [L + c + R[1:] for L, R in self._splits(word) for c in self.alphabet if R]

    def _edit_one_letter(self, word, allow_switches=True):
        edit_one_set = set(self._deletes(word))
        edit_one_set.update(set(self._inserts(word)))
        edit_one_set.update(set(self._replaces(word)))
        if allow_switches:
            edit_one_set.update(set(self._switches(word)))

        return edit_one_set

    def _edit_two_letters(self, word, allow_switches=True):
        edit_two_set = set()
        for w in self._edit_one_letter(word, allow_switches=allow_switches):
            edit_two_set.update(self._edit_one_letter(w, allow_switches=allow_switches))

        return edit_two_set

    def _known(self, words):
        return set(w for w in words if w in self.vocab)

    def get_corrections(self, word, n=2, allow_switches=True, include_probs=True):
        suggestions = (self._known([word]) or
                       self._known(self._edit_one_letter(word, allow_switches=allow_switches)) or
                       self._known(self._edit_two_letters(word, allow_switches=allow_switches)) or
                       [word])
        n_best = sorted([(w, self.probs.get(w, 0)) for w in suggestions],
                        key=lambda x: x[1],
                        reverse=True)[:n]
        if not include_probs:
            n_best = [w for w, prob in n_best]

        return n_best

    # TODO: compete the auto_correct function
    # def auto_correct(self, sentence):
    #    return [self.get_corrections(word, n=1, include_probs=False) for word in re.findall(r'\w+', sentence)]


