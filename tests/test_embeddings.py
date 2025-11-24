import pytest
import numpy as np
from autoannotate.core.embeddings import EmbeddingExtractor


@pytest.fixture
def sample_timeseries():
    np.random.seed(42)
    series_list = [
        np.random.randn(100) + np.linspace(0, 1, 100),
        np.random.randn(100) + np.linspace(1, 0, 100),
        np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.randn(100) * 0.1,
        np.cos(np.linspace(0, 4 * np.pi, 100)) + np.random.randn(100) * 0.1,
        np.random.randn(100),
    ]
    return [s.astype(np.float32) for s in series_list]


class TestEmbeddingExtractor:

    def test_initialization(self):
        extractor = EmbeddingExtractor(
            model_name="chronos-t5-tiny", batch_size=16, context_length=256
        )

        assert extractor.model_name == "chronos-t5-tiny"
        assert extractor.batch_size == 16
        assert extractor.context_length == 256

    def test_extract_embeddings(self, sample_timeseries):
        extractor = EmbeddingExtractor(
            model_name="chronos-t5-tiny", batch_size=2, context_length=128
        )

        embeddings = extractor(sample_timeseries)

        assert embeddings.shape[0] == len(sample_timeseries)
        assert embeddings.shape[1] > 0
        assert not np.isnan(embeddings).any()

    def test_preprocess_series_padding(self):
        extractor = EmbeddingExtractor(model_name="chronos-t5-tiny", context_length=128)

        short_series = np.random.randn(50).astype(np.float32)
        preprocessed = extractor._preprocess_series(short_series)

        assert preprocessed.shape[0] == 128

    def test_preprocess_series_truncation(self):
        extractor = EmbeddingExtractor(model_name="chronos-t5-tiny", context_length=128)

        long_series = np.random.randn(200).astype(np.float32)
        preprocessed = extractor._preprocess_series(long_series)

        assert preprocessed.shape[0] == 128

    def test_invalid_model_name(self):
        with pytest.raises(ValueError):
            EmbeddingExtractor(model_name="invalid-model")
