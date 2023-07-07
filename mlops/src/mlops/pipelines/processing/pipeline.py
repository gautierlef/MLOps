from kedro.pipeline import Pipeline, node
from .nodes import split_dataset


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                clean_dataset,
                [],
                dict(
                    X_train="X_train",
                    y_train="y_train",
                    X_test="X_test",
                    y_test="y_test"
                )
            )
        ]
    )