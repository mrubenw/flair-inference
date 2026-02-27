import pytest
import torch
import flair

from flair.data import Sentence
from flair.datasets import ClassificationCorpus
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.nn import DeepNCMDecoder
from tests.model_test_utils import BaseModelTest


class TestDeepNCMDecoder(BaseModelTest):
    model_cls = TextClassifier
    train_label_type = "class"
    multiclass_prediction_labels = ["POSITIVE", "NEGATIVE"]
    training_args = {
        "max_epochs": 2,
        "mini_batch_size": 4,
        "learning_rate": 1e-5,
    }

    @pytest.fixture()
    def embeddings(self):
        return TransformerDocumentEmbeddings("distilbert-base-uncased", fine_tune=True)

    @pytest.fixture()
    def corpus(self, tasks_base_path):
        return ClassificationCorpus(tasks_base_path / "imdb", label_type=self.train_label_type)

    @pytest.fixture()
    def multiclass_train_test_sentence(self):
        return Sentence("This movie was great!")

    def build_model(self, embeddings, label_dict, **kwargs):

        model_args = {
            "embeddings": embeddings,
            "label_dictionary": label_dict,
            "label_type": self.train_label_type,
            "use_encoder": False,
            "encoding_dim": 64,
            "alpha": 0.95,
            "mean_update_method": "online",
        }
        model_args.update(kwargs)

        deepncm_decoder = DeepNCMDecoder(
            label_dictionary=model_args["label_dictionary"],
            embeddings_size=model_args["embeddings"].embedding_length,
            alpha=model_args["alpha"],
            encoding_dim=model_args["encoding_dim"],
            mean_update_method=model_args["mean_update_method"],
        )

        model = self.model_cls(
            embeddings=model_args["embeddings"],
            label_dictionary=model_args["label_dictionary"],
            label_type=model_args["label_type"],
            multi_label=model_args.get("multi_label", False),
            decoder=deepncm_decoder,
        )

        return model

    @pytest.mark.integration()
    @pytest.mark.skip(reason="flair.trainers removed from inference-only build")
    def test_train_load_use_classifier(
        self, results_base_path, corpus, embeddings, example_sentence, train_test_sentence
    ):
        pass  # Trainer tests skipped - flair.trainers removed

    def test_get_prototype(self, corpus, embeddings):
        label_dict = corpus.make_label_dictionary(label_type=self.train_label_type)
        model = self.build_model(embeddings, label_dict)

        # Get the first *actual* label name, skipping '<unk>' if present
        valid_label_items = [item for item in label_dict.get_items() if item != "<unk>"]
        assert (
            valid_label_items
        ), "Label dictionary should contain labels other than <unk>"  # Ensure there are actual labels
        first_valid_item = valid_label_items[0]

        prototype = model.decoder.get_prototype(first_valid_item)  # Use the first valid item
        assert isinstance(prototype, torch.Tensor)
        assert prototype.shape == (model.decoder.encoding_dim,)

        with pytest.raises(ValueError):
            model.decoder.get_prototype("NON_EXISTENT_CLASS")

    def test_get_closest_prototypes(self, corpus, embeddings):
        label_dict = corpus.make_label_dictionary(label_type=self.train_label_type)
        model = self.build_model(embeddings, label_dict)
        input_vector = torch.randn(model.decoder.encoding_dim, device=flair.device)
        closest_prototypes = model.decoder.get_closest_prototypes(input_vector, top_k=2)

        assert len(closest_prototypes) == 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in closest_prototypes)

        with pytest.raises(ValueError):
            error_vector = torch.randn(model.decoder.encoding_dim + 1, device=flair.device)
            model.decoder.get_closest_prototypes(error_vector)

    def test_forward_loss(self, corpus, embeddings):
        label_dict = corpus.make_label_dictionary(label_type=self.train_label_type)
        model = self.build_model(embeddings, label_dict)

        sentences = [Sentence("This movie was great!"), Sentence("I didn't enjoy this film at all.")]
        for sentence, label in zip(sentences, list(label_dict.get_items())[:2]):
            sentence.add_label(self.train_label_type, label)

        loss, count = model.forward_loss(sentences)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert count == len(sentences)

    @pytest.mark.parametrize("mean_update_method", ["online", "condensation", "decay"])
    def test_mean_update_methods(self, corpus, embeddings, mean_update_method):
        label_dict = corpus.make_label_dictionary(label_type=self.train_label_type)
        model = self.build_model(embeddings, label_dict, mean_update_method=mean_update_method)

        initial_prototypes = model.decoder.class_prototypes.clone()

        sentences = [Sentence("This movie was great!"), Sentence("I didn't enjoy this film at all.")]
        for sentence, label in zip(sentences, list(label_dict.get_items())[:2]):
            sentence.add_label(self.train_label_type, label)

        model.forward_loss(sentences)
        model.decoder.update_prototypes()

        assert not torch.all(torch.eq(initial_prototypes, model.decoder.class_prototypes))

    @pytest.mark.parametrize("mean_update_method", ["online", "condensation", "decay"])
    @pytest.mark.skip(reason="flair.trainers removed from inference-only build")
    def test_deepncm_plugin(self, corpus, embeddings, mean_update_method):
        pass  # Trainer tests skipped - flair.trainers removed
