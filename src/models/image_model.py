"""ResNet50 transfer-learning image regressor."""

from ._tensorflow import get_tensorflow


def build_image_model(
    image_size: int = 224,
    weights: str | None = "imagenet",
    trainable_backbone: bool = False,
    dropout_rate: float = 0.2,
):
    """Build a ResNet50 feature extractor with a scalar regression head."""
    if image_size <= 0 or not 0 <= dropout_rate < 1:
        raise ValueError("image_size must be positive and dropout_rate must be in [0, 1).")
    tf = get_tensorflow()
    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights=weights,
        input_shape=(image_size, image_size, 3),
    )
    backbone.trainable = trainable_backbone
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="galaxy_image")
    features = backbone(inputs, training=trainable_backbone)
    features = tf.keras.layers.GlobalAveragePooling2D(name="image_pooling")(features)
    features = tf.keras.layers.Dense(128, activation="relu", name="image_dense")(features)
    if dropout_rate:
        features = tf.keras.layers.Dropout(dropout_rate)(features)
    output = tf.keras.layers.Dense(1, name="redshift")(features)
    return tf.keras.Model(inputs=inputs, outputs=output, name="resnet50_redshift")

