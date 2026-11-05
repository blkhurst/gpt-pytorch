"""UCI Tokenizer (Universal Chess Interface)"""


class UCITokenizer:
    def __init__(self):
        # Special tokens mark padding and game boundaries
        tokens = ["<|pad|>", "<|bos|>", "<|eos|>"]
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

        # Every pair of different squares, including castling and captures
        files = "abcdefgh"
        squares = [f"{file}{rank}" for rank in "12345678" for file in files]
        for origin in squares:
            for destination in squares:
                if origin != destination:
                    tokens.append(origin + destination)

        # Promotions forward or diagonal, for either colour, to Q/R/B/N
        for origin_rank, destination_rank in [("7", "8"), ("2", "1")]:
            for origin_file, origin in enumerate(files):
                for destination_file, destination in enumerate(files):
                    if abs(origin_file - destination_file) <= 1:
                        for piece in "qrbn":
                            tokens.append(
                                f"{origin}{origin_rank}{destination}{destination_rank}{piece}"
                            )

        # Token IDs
        self.vocab = {token: index for index, token in enumerate(tokens)}
        self.tokens = tokens
        self.vocab_size = len(tokens)

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        ids = []
        for move in text.split():
            if move not in self.vocab:
                raise ValueError(f"Unknown UCI token: {move!r}")
            ids.append(self.vocab[move])
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for token_id in ids:
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"Unknown token ID: {token_id}")
            if skip_special_tokens and token_id < 3:
                continue
            tokens.append(self.tokens[token_id])
        return " ".join(tokens)
