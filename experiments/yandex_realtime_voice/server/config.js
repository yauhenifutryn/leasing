export function loadConfig(env) {
  const required = ["YC_FOLDER_ID"];
  for (const key of required) {
    if (!env[key]) throw new Error(`Missing ${key}`);
  }
  const apiKey = env.YC_API_KEY;
  const iamToken = env.YC_IAM_TOKEN;
  if (!apiKey && !iamToken) throw new Error("Missing YC_API_KEY or YC_IAM_TOKEN");

  const model = env.YC_MODEL || `gpt://${env.YC_FOLDER_ID}/speech-realtime-250923`;
  const wsUrl =
    env.YC_REALTIME_WS_URL ||
    `wss://ai.api.cloud.yandex.net/v1/realtime/openai?model=${encodeURIComponent(model)}`;

  return {
    folderId: env.YC_FOLDER_ID,
    apiKey,
    iamToken,
    model,
    wsUrl,
    voice: env.YC_VOICE || "ermil",
    vectorStoreId: env.YC_VECTOR_STORE_ID || "",
  };
}
