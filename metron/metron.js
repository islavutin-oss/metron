// Copyright (c) 2024 AXELTEC SOFTWARE LTD.
// SPDX-License-Identifier: Apache-2.0

import http from 'k6/http';
import { check } from 'k6';
import { SharedArray } from 'k6/data';
import { Trend, Counter } from 'k6/metrics';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.2/index.js';
import exec from 'k6/execution';
import { generateOpenAIPayload } from './requests.js';


const ttftMetric = new Trend('ttft', true);
const itlMetric = new Trend('itl', true);
const tpotMetric = new Trend('tpot', true);
const generated_tokens_per_request = new Trend('number_of_generated_tokens', false);
const total_generated_tokens = new Counter('total_generated_tokens')
const total_requests_sent = new Counter('total_requests_sent')
const responses_without_content = new Counter('responses_without_content')
const token_counts_estimated = new Counter('token_counts_estimated')
const incomplete_streams = new Counter('incomplete_streams')
const itlPerTokenMetric = new Trend('itl_per_token', true);

let k6_scenarios;

if (__ENV.EXECUTOR == 'constant-arrival-rate') {
    k6_scenarios = {
        streaming: {
            executor: __ENV.EXECUTOR,
            rate: __ENV.RPS,
            timeUnit: '1s',
            duration: __ENV.DURATION,
            preAllocatedVUs: Math.min(parseInt(__ENV.MAX_VUS) || 500, 500),
            maxVUs: parseInt(__ENV.MAX_VUS) || 500,
        },
    }
} else {
    if (__ENV.EXECUTOR == 'constant-vus') {
        k6_scenarios = {
            streaming: {
                executor: __ENV.EXECUTOR,
                vus: __ENV.VUS,
                duration: __ENV.DURATION
            },
        }
    } else {
        k6_scenarios = {
            streaming: {
                executor: __ENV.EXECUTOR,
                vus: __ENV.VUS,
                iterations: __ENV.ITERATIONS
            },
        }
    }
}

export const options = {
    scenarios: k6_scenarios
};

// Deterministic per-VU stream so a run is reproducible from its seed.
const seed = parseInt(__ENV.SEED) || Math.floor(Date.now() / 1000);

function makeRng(a) {
    return function () {
        a |= 0; a = (a + 0x6D2B79F5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

const url = `http://localhost:${__ENV.PROXY_PORT}${__ENV.API_ROUTE}`;

const prompts = new SharedArray('prompts', function () {
    const artifact = __ENV.ARTIFACT;
    const f = JSON.parse(open(artifact.concat('/prompts.json')));
    return f;
});

const gen_lengths = new SharedArray('gen_lengths', function () {
    const artifact = __ENV.ARTIFACT;
    const f = JSON.parse(open(artifact.concat('/gen_lengths.json')));
    return f;
});

export default function () {
    const rng = makeRng(seed + exec.scenario.iterationInTest);
    const random_prompt_id = Math.floor(rng() * prompts.length);
    const randomPrompt = prompts[random_prompt_id];
    const randomMaxTokens = gen_lengths[Math.floor(rng() * gen_lengths.length)];
    const payload = generateOpenAIPayload(randomPrompt, randomMaxTokens);

    const bearer_token = "Bearer".concat(" ", __ENV.TOKEN);

    total_requests_sent.add(1)

    const response = http.post(url, payload, {
        headers: {
            'Content-Type': 'application/json',
            'Authorization': bearer_token
        },
    });

    check(response, {
        'is status 200': (r) => r.status === 200,
    });

    // A non-2xx produced no tokens, whatever its body says. Counting anything
    // for it — least of all the requested max_tokens via the fallback below —
    // reports a failed run as a fast one. Skip it; run health surfaces the
    // failures separately.
    const ok = response.status >= 200 && response.status < 300;
    let generated_tokens_filled = false;

    if (ok && response.body) {
        try {
            const responseData = JSON.parse(response.body);

            const e2e = responseData.e2e || null;
            const ttft = responseData.ttft || null;
            const itlArray = responseData.itl || [];
            const completion_tokens = responseData.completion_tokens ?? null;
            const content_chunks = responseData.content_chunks ?? null;
            let total_tokens = -1;

            if (responseData.complete === false) {
                incomplete_streams.add(1);
            }

            if (ttft !== null) {
                ttftMetric.add(ttft);
            }

            if (completion_tokens !== null){
                total_generated_tokens.add(completion_tokens);
                generated_tokens_per_request.add(completion_tokens);
                total_tokens = completion_tokens;
                generated_tokens_filled = true;
            }

            if (itlArray.length > 0) {
                for (const itl of itlArray) {
                    itlMetric.add(itl);
                }
                if (generated_tokens_filled === false && content_chunks) {
                    total_generated_tokens.add(content_chunks);
                    generated_tokens_per_request.add(content_chunks);
                    total_tokens = content_chunks;
                    generated_tokens_filled = true;
                    token_counts_estimated.add(1);
                }
            }

            if (content_chunks && total_tokens > content_chunks) {
                const tokens_per_chunk = total_tokens / content_chunks;
                for (const itl of itlArray) {
                    itlPerTokenMetric.add(itl / tokens_per_chunk);
                }
            }

            // total_tokens > 1 or the divisor is zero: a one-token completion
            // has no inter-token interval, and dividing produced +Inf.
            if (e2e != null && ttft !== null && generated_tokens_filled && total_tokens > 1) {
                tpotMetric.add((e2e - ttft) / (total_tokens - 1));
            }

            if (responseData?.response?.usage?.completion_tokens !== undefined && generated_tokens_filled === false) {
                total_generated_tokens.add(responseData.response.usage.completion_tokens);
                generated_tokens_per_request.add(responseData.response.usage.completion_tokens);
                generated_tokens_filled = true;
            }

        } catch (error) {
            console.error('Error parsing JSON response:', error);
        }

        if (generated_tokens_filled === false) {
            responses_without_content.add(1);
        }
    } else {
        console.error('Response body is null or undefined.');
    }
}


export function handleSummary(data) {

    const requestsSent = data.metrics['total_requests_sent'];

    delete data.metrics['http_req_duration{expected_response:true}'];
    delete data.metrics['checks'];

    data.metrics['end-to-end_latency'] = data.metrics['http_req_duration'];

    for (const key in data.metrics) {
        if (key.startsWith('iteration') || key.startsWith('data') || key.startsWith('http')) {
            delete data.metrics[key];
        }
    }

    // total_requests_sent is how many requests this level actually completed,
    // and it is the n behind every percentile above. It used to be deleted
    // here, alongside http_reqs (which the http* sweep above removes), which
    // left the summary with ten percentiles and no sample size: the reader
    // could not tell an earned p99 from one taken off a hundred observations,
    // and neither could metron -- it reports a percentile as unsupported when
    // it cannot count the samples, so every p90 and p99 printed as an em dash.
    // Kept last, after the sweep that would otherwise take it.
    data.metrics['total_requests_sent'] = requestsSent;

    return {
      stdout: textSummary(data, { indent: ' ', enableColors: true }),
    };
}
