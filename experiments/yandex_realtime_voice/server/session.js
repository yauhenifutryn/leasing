export function buildSessionUpdate({ voice, vectorStoreId }) {
  const session = {
    voice,
    modalities: ["text", "audio"],
    input_audio_format: "pcm16",
    output_audio_format: "pcm16",
    tool_choice: vectorStoreId ? "auto" : "none",
    tools: vectorStoreId
      ? [
          {
            type: "file_search",
            vector_store_ids: [vectorStoreId],
            max_num_results: 3,
          },
        ]
      : [],
  };
  return { type: "session.update", session };
}
