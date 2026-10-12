"""Two-input ResNet50 and photometric feature fusion regressor."""

from ._tensorflow import get_tensorflow


def build_multimodal_model(
    photometric_input_dim: int,
    image_size: int = 224,
    weights: str | None = "imagenet",
    trainable_backbone: bool = False,
    dropout_rate: float = 0.2,
):
    """Build a fused image+photometry model with a single redshift output."""
    if photometric_input_dim <= 0 or image_size <= 0:
        raise ValueError("Input dimensions must be positive.")
    if not 0 <= dropout_rate < 1:
        raise ValueError("dropout_rate must be in [0, 1).")
    tf = get_tensorflow()
    image_input = tf.keras.Input(
        shape=(image_size, image_size, 3), name="galaxy_image"
    )
    photometric_input = tf.keras.Input(
        shape=(photometric_input_dim,), name="photometric_features"
    )
    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights=weights,
        input_shape=(image_size, image_size, 3),
    )
    backbone.trainable = trainable_backbone
    image_features = backbone(image_input, training=trainable_backbone)
    image_features = tf.keras.layers.GlobalAveragePooling2D(name="image_pooling")(
        image_features
    )
    photo_features = tf.keras.layers.Dense(
        64, activation="relu", name="photometric_dense"
    )(photometric_input)
    if dropout_rate:
        photo_features = tf.keras.layers.Dropout(dropout_rate)(photo_features)
    fused = tf.keras.layers.Concatenate(name="feature_fusion")(
        [image_features, photo_features]
    )
    fused = tf.keras.layers.Dense(128, activation="relu", name="joint_dense")(fused)
    if dropout_rate:
        fused = tf.keras.layers.Dropout(dropout_rate)(fused)
    output = tf.keras.layers.Dense(1, name="redshift")(fused)
    return tf.keras.Model(
        inputs=[image_input, photometric_input],
        outputs=output,
        name="multimodal_redshift",
    )

