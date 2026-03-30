use embed_anything::embeddings::embed::EmbedderBuilder;

fn main() {
    let _ = EmbedderBuilder::new()
        .model_architecture("bert")
        .onnx_model_path(Some("model.onnx"))
        .tokenizer_path(Some("tokenizer.json"))
        .from_local_onnx();
}
