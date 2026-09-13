// Copyright (c) 2024 AXELTEC SOFTWARE LTD.
// SPDX-License-Identifier: Apache-2.0

// The stopwatch.
//
// k6 drives the load but cannot see a token arrive: its http module returns the
// response body as a completed string, so there is no way to timestamp the
// chunks of a stream from inside a k6 script. This process sits between them,
// reads the SSE stream as it arrives, and records when each token appeared.
// That is the measurement metron exists to take; the proxy is the instrument,
// not glue.
//
// Node's built-in http is all it needs — one route, forwarding a stream. It
// used to run on express and axios, which meant an npm install, two supply
// chain dependencies, and a request body parsed into an object then serialised
// back again on the critical path. None of that bought anything.
//
// Where the clock starts matters: startTime is taken once the request body has
// been received, so the k6-to-proxy hop is outside every number reported. What
// remains inside is the proxy-to-server call, the server itself, and Node's
// scheduling delay before the timestamp is taken.

const http = require('http');
const https = require('https');
const { performance } = require('perf_hooks');
const { URL } = require('url');

const PORT = parseInt(process.env.PROXY_PORT) || 3000;
const TARGET = new URL(process.env.URL);
const client = TARGET.protocol === 'https:' ? https : http;
const ROUTE = process.env.API_ROUTE;
const STREAMING = process.env.STREAM === 'true';

// Connections are reused, so a sweep does not pay for a TCP handshake per
// request and attribute it to the server.
const agent = new client.Agent({ keepAlive: true, maxSockets: Infinity });

const upstreamHeaders = () => {
    const h = { 'Content-Type': 'application/json' };
    if (process.env.TOKEN) h['Authorization'] = `Bearer ${process.env.TOKEN}`;
    if (STREAMING) h['Accept'] = 'text/event-stream';
    return h;
};

const sendJson = (res, status, obj) => {
    const payload = JSON.stringify(obj);
    res.writeHead(status, {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(payload),
    });
    res.end(payload);
};

const handleStream = (body, res, startTime) => {
    let firstTokenTime = null;
    let lastTokenTime = null;
    let ttft = null;
    let completionTokensValue = -1;
    let upstreamStatus = 0;
    let contentChunks = 0;
    let sawTerminator = false;
    const requestItlArray = [];

    // Reply to k6 exactly once. Without this a stream that ends without a
    // [DONE] (a server that errors or closes early) would never answer, and the
    // k6 request would hang until its timeout — inflating the tail with a
    // failure disguised as a slow success.
    let responded = false;
    const reply = () => {
        if (responded) return;
        responded = true;
        const ok = upstreamStatus >= 200 && upstreamStatus < 300;
        sendJson(res, ok ? 200 : upstreamStatus || 502, {
            e2e: performance.now() - startTime,
            ttft: ttft,
            itl: requestItlArray,
            upstream_status: upstreamStatus,
            content_chunks: contentChunks,
            complete: sawTerminator,
            ...(completionTokensValue !== -1 && { completion_tokens: completionTokensValue }),
        });
    };

    const upstream = client.request(
        { method: 'POST', hostname: TARGET.hostname, port: TARGET.port,
          path: TARGET.pathname + TARGET.search, headers: upstreamHeaders(), agent },
        (up) => {
            upstreamStatus = up.statusCode;
            up.setEncoding('utf8');
            let carry = '';
            up.on('data', (chunk) => {
                // SSE events can split across TCP reads; keep the tail until a
                // newline completes it, or a token is lost and its interval
                // silently merges with the next one.
                carry += chunk;
                const lines = carry.split('\n');
                carry = lines.pop();
                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    const eventData = line.slice(6).trim();
                    if (eventData === '[DONE]') { sawTerminator = true; reply(); continue; }
                    try {
                        const tokenData = JSON.parse(eventData);
                        const responseField = ROUTE === '/v1/chat/completions'
                            ? tokenData.choices[0]?.delta?.content ||
                              tokenData.choices[0]?.message?.content ||
                              tokenData.choices[0]?.content
                            : tokenData.choices[0]?.text;

                        if (tokenData.usage?.completion_tokens !== undefined) {
                            completionTokensValue = tokenData.usage.completion_tokens;
                        }

                        // Truthy, not `!== ""`: the first SSE chunk is often a
                        // role-only delta with no content field. Counting it set
                        // TTFT on an empty chunk and added a phantom interval.
                        if (tokenData.choices?.[0]?.finish_reason) sawTerminator = true;

                        if (responseField) {
                            contentChunks += 1;
                            const currentTime = performance.now();
                            if (!firstTokenTime) {
                                firstTokenTime = currentTime;
                                ttft = firstTokenTime - startTime;
                            }
                            if (lastTokenTime) requestItlArray.push(currentTime - lastTokenTime);
                            lastTokenTime = currentTime;
                        }
                    } catch (error) {
                        console.error('Error parsing token:', error.message);
                    }
                }
            });
            up.on('end', reply);
            up.on('error', (err) => {
                console.error('Stream error:', err.message);
                if (!responded) { responded = true; sendJson(res, 502, { error: 'upstream stream error' }); }
            });
        }
    );

    upstream.on('error', (err) => {
        console.error('Error in streaming request:', err.message);
        if (!responded) { responded = true; sendJson(res, 502, { error: 'upstream error' }); }
    });
    upstream.end(body);
};

const handleNonStream = (body, res) => {
    const upstream = client.request(
        { method: 'POST', hostname: TARGET.hostname, port: TARGET.port,
          path: TARGET.pathname + TARGET.search, headers: upstreamHeaders(), agent },
        (up) => {
            const parts = [];
            up.on('data', (c) => parts.push(c));
            up.on('end', () => {
                let parsed;
                try { parsed = JSON.parse(Buffer.concat(parts).toString()); }
                catch { parsed = Buffer.concat(parts).toString(); }
                const ok = up.statusCode >= 200 && up.statusCode < 300;
                sendJson(res, ok ? 200 : up.statusCode, {
                    response: parsed,
                    upstream_status: up.statusCode,
                });
            });
        }
    );
    upstream.on('error', (err) => {
        console.error('Error in non-streaming request:', err.message);
        sendJson(res, 500, { error: 'internal server error' });
    });
    upstream.end(body);
};

const server = http.createServer((req, res) => {
    if (req.method !== 'POST' || req.url.split('?')[0] !== ROUTE) {
        sendJson(res, 404, { error: 'not found' });
        return;
    }
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
        // Clock starts here: the body is in hand, nothing of k6's hop is counted.
        const startTime = performance.now();
        const body = Buffer.concat(chunks);
        if (STREAMING) handleStream(body, res, startTime);
        else handleNonStream(body, res);
    });
});

server.listen(PORT, () => {
    console.log(`Proxy server listening on port ${PORT}`);
});
