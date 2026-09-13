// Copyright (c) 2024 AXELTEC SOFTWARE LTD.
// SPDX-License-Identifier: Apache-2.0
export function generateOpenAIPayload(randomPrompt, randomMaxTokens) {
    const isChatCompletions = __ENV.API_ROUTE.includes("chat/completions");

    return JSON.stringify({
        model: __ENV.MODEL_NAME,
        ...(isChatCompletions ? {
            messages: [
                {
                    role: "user",
                    content: randomPrompt,
                },
            ],
            ...(__ENV.LOGPROBS == "true" ? {
                top_logprobs: parseInt(__ENV.TOP_LOGPROBS),
                logprobs: true,
            } : {
                logprobs: false,
            }),
        } : {
            prompt: randomPrompt,
            logprobs: parseInt(__ENV.LOGPROBS),
        }),
        max_tokens: randomMaxTokens,
        temperature: parseFloat(__ENV.TEMPERATURE),
        top_p: parseFloat(__ENV.TOP_P),
        n: parseInt(__ENV.NUM_SEQUENCES),
        stream: __ENV.STREAM === "true",
        top_k: parseInt(__ENV.TOP_K),
        ...(__ENV.STREAM === "true" && __ENV.STREAM_OPTIONS ? {
            stream_options: JSON.parse(__ENV.STREAM_OPTIONS)
        } : {})
    });
}

