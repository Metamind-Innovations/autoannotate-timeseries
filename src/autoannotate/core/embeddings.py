import torch
import torch.nn.functional as F
from typing import List, Literal, Optional
import numpy as np
from tqdm import tqdm


class EmbeddingExtractor:

    def __init__(
        self,
        model_name: Literal["chronos-t5-tiny", "chronos-t5-small"] = "chronos-t5-tiny",
        device: Optional[str] = None,
        batch_size: int = 32,
        context_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.context_length = context_length
        self._load_model()

    def _load_model(self):
        if self.model_name == "chronos-t5-tiny":
            from chronos import ChronosPipeline

            self.model = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-tiny",
                device_map=self.device,
                torch_dtype=torch.float32,
            )
        elif self.model_name == "chronos-t5-small":
            from chronos import ChronosPipeline

            self.model = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-small",
                device_map=self.device,
                torch_dtype=torch.float32,
            )
        else:
            raise ValueError(
                f"Unknown model: {self.model_name}. Available: chronos-t5-tiny, chronos-t5-small"
            )

        if hasattr(self.model, "eval"):
            self.model.eval()

    def _preprocess_series(self, series: np.ndarray) -> np.ndarray:
        if len(series) > self.context_length:
            series = series[-self.context_length :]
        elif len(series) < self.context_length:
            pad_width = self.context_length - len(series)
            series = np.pad(series, (pad_width, 0), mode="edge")

        return series.astype(np.float32)

    def extract_single(self, series: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            series_processed = self._preprocess_series(series)
            series_tensor = torch.FloatTensor(series_processed).unsqueeze(0).to(self.device)

            result = self.model.embed(series_tensor)
            if isinstance(result, tuple):
                embedding = result[0]
            else:
                embedding = result
            embedding = embedding.mean(dim=1)

            embedding = F.normalize(embedding, p=2, dim=-1)
            return embedding.cpu().numpy().flatten()

    def extract_batch(self, series_list: List[np.ndarray]) -> np.ndarray:
        embeddings = []

        for i in tqdm(range(0, len(series_list), self.batch_size), desc="Extracting embeddings"):
            batch = series_list[i : i + self.batch_size]

            batch_processed = [self._preprocess_series(s) for s in batch]
            batch_array = np.array(batch_processed)
            batch_tensor = torch.FloatTensor(batch_array).to(self.device)

            with torch.no_grad():
                result = self.model.embed(batch_tensor)
                if isinstance(result, tuple):
                    batch_embeddings = result[0]
                else:
                    batch_embeddings = result
                batch_embeddings = batch_embeddings.mean(dim=1)

                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=-1)
                embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    def __call__(self, series_list: List[np.ndarray]) -> np.ndarray:
        return self.extract_batch(series_list)
