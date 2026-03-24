import time

class WordBuilder:
    def __init__(self):
        self.current_word = ""
        self.sentence = ""
        self.current_letter = None
        self.stable_count = 0
        self.letter_committed = False
        self.last_commit_time = 0
        self.letter_buffer = []  # stores last 15 predictions

        # Tuning
        self.confidence_threshold = 0.75
        self.required_stable_frames = 15
        self.commit_cooldown = 2.0

        # Letters that get confused — map wrong → right
        self.confusion_map = {
            'C': self._resolve_c,
        }

    def _resolve_c(self, buffer):
        """
        If model says C but O appears frequently
        in recent buffer → it is probably O
        """
        letters = [b['letter'] for b in buffer[-15:]]
        o_count = letters.count('O')
        c_count = letters.count('C')

        if o_count >= 3:
            return 'O'
        return 'C'

    def add_to_buffer(self, letter, confidence):
        self.letter_buffer.append({
            'letter': letter,
            'confidence': confidence
        })
        if len(self.letter_buffer) > 30:
            self.letter_buffer.pop(0)

    def resolve_letter(self, letter):
        """Apply confusion map if needed"""
        if letter in self.confusion_map:
            return self.confusion_map[letter](self.letter_buffer)
        return letter

    def update(self, letter, confidence):
        now = time.time()

        # Add raw prediction to buffer
        self.add_to_buffer(letter, confidence)

        # Ignore low confidence
        if confidence < self.confidence_threshold:
            self.stable_count = 0
            self.current_letter = None
            self.letter_committed = False
            return self.get_state()

        # Resolve confused letters
        resolved_letter = self.resolve_letter(letter)

        # New letter started
        if resolved_letter != self.current_letter:
            self.current_letter = resolved_letter
            self.stable_count = 1
            self.letter_committed = False
            return self.get_state()

        # Same letter — increment stability
        self.stable_count += 1

        time_since_commit = now - self.last_commit_time
        can_commit = (
            self.stable_count >= self.required_stable_frames
            and not self.letter_committed
            and time_since_commit > self.commit_cooldown
        )

        if can_commit:
            self.letter_committed = True
            self.last_commit_time = now

            if resolved_letter == "space":
                if self.current_word:
                    self.sentence += self.current_word + " "
                    self.current_word = ""

            elif resolved_letter == "del":
                if self.current_word:
                    self.current_word = self.current_word[:-1]
                elif self.sentence:
                    self.sentence = self.sentence.rstrip()
                    if self.sentence:
                        self.sentence = self.sentence[:-1] + " "

            elif resolved_letter == "nothing":
                pass

            else:
                self.current_word += resolved_letter.upper()

        return self.get_state()

    def get_state(self):
        return {
            "current_word": self.current_word,
            "sentence": self.sentence.strip(),
            "last_letter": self.current_letter,
            "stable_frames": self.stable_count,
            "required_frames": self.required_stable_frames,
            "letter_committed": self.letter_committed
        }

    def clear(self):
        self.current_word = ""
        self.sentence = ""
        self.current_letter = None
        self.stable_count = 0
        self.letter_committed = False
        self.letter_buffer = []