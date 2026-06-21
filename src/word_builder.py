from collections import deque, Counter


class WordBuilder:
    def __init__(self):
        self.text = ""
        self.buffer = deque(maxlen=10)
        self.last_char = None
        self.frame_count = 0
        self.cooldown = 30

    def set_buffer_size(self, size):
        self.buffer = deque(list(self.buffer), maxlen=size)

    def set_cooldown(self, cooldown):
        self.cooldown = cooldown

    def update(self, prediction):

        self.buffer.append(prediction)

        if len(self.buffer) < self.buffer.maxlen:
            return self.text

        stable_char = Counter(self.buffer).most_common(1)[0][0]

        if stable_char == self.last_char:
            self.frame_count += 1
        else:
            self.last_char = stable_char
            self.frame_count = 0

        if self.frame_count < self.cooldown:
            return self.text

        self.frame_count = 0

        if stable_char == "space":
            if not self.text.endswith(" "):
                self.text += " "

        elif stable_char == "del":
            self.text = self.text[:-1]

        else:
            self.text += stable_char

        return self.text