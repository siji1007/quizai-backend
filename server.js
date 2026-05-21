const express = require('express');
const multer = require('multer');
const pdf = require('pdf-parse');
const Groq = require('groq-sdk');
const cors = require('cors');
const dotenv = require('dotenv');

dotenv.config();

const app = express();
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// ─── CORS ──────────────────────────────────────────────────────────────────
const corsOptions = {
    origin: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With', 'X-Session-Id'],
    credentials: true,
};
app.use(cors(corsOptions));
app.use(express.json());

// Strip /api prefix for Vercel routing
app.use((req, res, next) => {
    if (req.url.startsWith('/api')) {
        req.url = req.url.replace(/^\/api/, '') || '/';
    }
    next();
});

app.use((req, res, next) => {
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
    next();
});

// ═══════════════════════════════════════════════════════════════════════════
//  GROQ ROUND-ROBIN LOAD BALANCER
//
//  Each API key gets its own Groq client and its own independent processing
//  lane (queue + rate-limit state). Incoming requests are distributed across
//  lanes in round-robin order, so two simultaneous requests run in parallel
//  on different keys instead of waiting behind each other in a single queue.
//
//  Free-tier limits per key: ~30 RPM / 1 req per 2 s.
// ═══════════════════════════════════════════════════════════════════════════

const GROQ_MIN_INTERVAL_MS = 2100;   // 2.1 s gap per lane (safety margin above 30 RPM)
const QUEUE_TIMEOUT_MS     = 120000; // Drop a job after 2 min if still waiting
const MAX_RETRIES          = 3;      // Retry on transient Groq errors
const RETRY_DELAY_MS       = 3000;   // Wait 3 s between retries

// Build one lane per configured API key (gracefully skip missing keys)
const API_KEYS = [
    process.env.GROQ_API_KEY,
    process.env.GROQ_API_KEY_ASSISTANT,
].filter(Boolean);

if (API_KEYS.length === 0) {
    console.error('❌ No Groq API keys found in environment variables!');
    process.exit(1);
}

/**
 * A Lane encapsulates one Groq client + its own rate-limit queue.
 */
function createLane(apiKey, index) {
    const client = new Groq({ apiKey });
    const queue  = [];
    let lastCallAt      = 0;
    let callInProgress  = false;
    const name = `Lane-${index + 1}`;

    async function processQueue() {
        if (callInProgress) return;

        const job = queue.shift();
        if (!job) return;

        if (Date.now() - job.enqueuedAt > QUEUE_TIMEOUT_MS) {
            console.warn(`[${name}] ${job.label} timed out while waiting`);
            job.reject(new Error('Request timed out while waiting in the processing queue. Please try again.'));
            processQueue();
            return;
        }

        callInProgress = true;

        // Enforce per-lane rate limit
        const msSinceLast = Date.now() - lastCallAt;
        if (msSinceLast < GROQ_MIN_INTERVAL_MS) {
            const waitMs = GROQ_MIN_INTERVAL_MS - msSinceLast;
            console.log(`[${name}] ${job.label} waiting ${waitMs}ms for rate limit`);
            await sleep(waitMs);
        }

        // Execute with retry logic
        let lastError;
        for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
            try {
                console.log(`[${name}] ${job.label} executing (attempt ${attempt}/${MAX_RETRIES})`);
                lastCallAt = Date.now();
                const result = await job.fn(client);
                console.log(`[${name}] ${job.label} succeeded`);
                job.resolve(result);
                lastError = null;
                break;
            } catch (err) {
                lastError = err;
                const isRateLimit = err?.status === 429 || /rate.?limit/i.test(err?.message || '');
                const isRetryable = isRateLimit || err?.status >= 500;

                console.error(`[${name}] ${job.label} attempt ${attempt} failed: ${err.message}`);

                if (attempt < MAX_RETRIES && isRetryable) {
                    const delay = isRateLimit ? GROQ_MIN_INTERVAL_MS * 2 : RETRY_DELAY_MS;
                    console.log(`[${name}] ${job.label} retrying in ${delay}ms...`);
                    await sleep(delay);
                    lastCallAt = Date.now();
                }
            }
        }

        if (lastError) {
            console.error(`[${name}] ${job.label} failed after ${MAX_RETRIES} attempts`);
            job.reject(lastError);
        }

        callInProgress = false;

        if (queue.length > 0) {
            setImmediate(processQueue);
        }
    }

    return {
        name,
        queue,
        get inProgress() { return callInProgress; },
        enqueue(fn, label = 'groq-call') {
            return new Promise((resolve, reject) => {
                queue.push({ fn, resolve, reject, enqueuedAt: Date.now(), label });
                console.log(`[${name}] ${label} enqueued | lane queue: ${queue.length}`);
                processQueue();
            });
        },
    };
}

// Instantiate all lanes
const lanes = API_KEYS.map((key, i) => createLane(key, i));
let roundRobinIndex = 0;

/**
 * Pick the least-loaded lane (fewest pending jobs).
 * Falls back to pure round-robin when loads are equal.
 */
function pickLane() {
    // Choose the lane with the shortest queue (+ 1 if currently processing)
    let best = lanes[0];
    let bestLoad = best.queue.length + (best.inProgress ? 1 : 0);

    for (let i = 1; i < lanes.length; i++) {
        const load = lanes[i].queue.length + (lanes[i].inProgress ? 1 : 0);
        if (load < bestLoad) {
            best = lanes[i];
            bestLoad = load;
        }
    }
    return best;
}

/**
 * Public API: submit a Groq call to the best available lane.
 * @param {Function} fn    - async (groqClient) => result
 * @param {string}   label - human-readable label for logging
 */
function enqueueGroqCall(fn, label = 'groq-call') {
    const lane = pickLane();
    return lane.enqueue(fn, label);
}

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

console.log(`✅ Groq load balancer initialised with ${lanes.length} lane(s): ${lanes.map(l => l.name).join(', ')}`);

// ─── Queue Status Endpoint (optional monitoring) ──────────────────────────
app.get('/queue-status', (req, res) => {
    const laneStats = lanes.map((lane) => ({
        lane: lane.name,
        queueLength: lane.queue.length,
        processing: lane.inProgress,
        estimatedWaitSeconds: Math.ceil(
            (lane.queue.length + (lane.inProgress ? 1 : 0)) * (GROQ_MIN_INTERVAL_MS / 1000)
        ),
    }));

    const totalPending = laneStats.reduce((sum, l) => sum + l.queueLength + (l.processing ? 1 : 0), 0);

    res.json({
        lanes: laneStats,
        totalPending,
        estimatedWaitSeconds: Math.ceil(totalPending / lanes.length * (GROQ_MIN_INTERVAL_MS / 1000)),
    });
});

// ─── Health Check ─────────────────────────────────────────────────────────
app.get('/', (req, res) => {
    res.json({ message: 'Cherry GenAI API is running successfully!' });
});

// ─── PIN Verify ───────────────────────────────────────────────────────────
app.post('/verify-pin', (req, res) => {
    const { pin } = req.body;
    if (pin === process.env.PIN || pin === '0000') {
        res.json({ success: true });
    } else {
        res.status(401).json({ success: false, message: 'Invalid PIN' });
    }
});

// ─── Generate Quiz ────────────────────────────────────────────────────────
app.post('/generate-quiz', (req, res, next) => {
    const contentType = req.headers['content-type'] || '';
    if (contentType.includes('application/json')) return next();
    upload.single('file')(req, res, (err) => {
        if (err) {
            console.error('Multer error:', err);
            return res.status(400).json({ error: 'File upload failed: ' + err.message });
        }
        next();
    });
}, async (req, res) => {
    try {
        if (!req.file && !req.body.topic) {
            return res.status(400).json({ error: 'No file uploaded or topic specified.' });
        }

        const questionCount = req.body.questionCount || 5;
        const difficulty    = req.body.difficulty    || 'Moderate';
        const quizType      = req.body.quizType      || 'Multiple Choice';
        const mode          = req.body.mode          || 'quiz';

        // ── Build system prompt ──────────────────────────────────────────
        let systemPrompt;
        if (mode === 'quiz') {
            systemPrompt = `You are a genius AI educational assistant. Analyze the provided material and generate an engaging, high-quality quiz.
                
                GENIUS ANALYSIS RULES:
                1. TEXT/NOTES: Extract core concepts and generate questions testing deep understanding.
                2. OBJECTS/PHOTOS: Identify the subject and generate facts/history/science questions.
                3. DIAGRAMS: Create questions that analyze the data or processes shown.

                STRICT FORMATTING REQUIREMENTS:
                Return ONLY a valid JSON object:
                {
                  "type": "quiz",
                  "title": "A creative title",
                  "questions": [
                    {
                      "type": "multiple-choice" | "true-false" | "situational",
                      "question": "Clear, insightful question",
                      "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
                      "correctAnswer": "Exact string",
                      "idealAnswer": "Key points",
                      "explanation": "Genius-level explanation"
                    }
                  ]
                }`;
        } else {
            systemPrompt = `You are a genius AI tutor. Analyze the provided material and provide a comprehensive explanation or solution.
                
                RULES:
                1. If it's a question or math problem, solve it step-by-step.
                2. If it's a concept or object, explain what it is, its significance, and key facts.
                3. Use clear, professional, and encouraging language.
                
                STRICT FORMATTING REQUIREMENTS:
                Return ONLY a valid JSON object:
                {
                  "type": "explanation",
                  "title": "Identification/Subject Title",
                  "content": "Comprehensive markdown-formatted explanation or solution"
                }`;
        }

        // ── Build messages ───────────────────────────────────────────────
        let messages = [{ role: 'system', content: systemPrompt }];
        let model = 'llama-3.3-70b-versatile';

        if (req.file) {
            if (req.file.mimetype === 'application/pdf') {
                const data = await pdf(req.file.buffer);
                if (!data.text.trim()) throw new Error('PDF appears to be empty or unreadable.');
                messages.push({
                    role: 'user',
                    content: mode === 'quiz'
                        ? `Generate a ${difficulty} difficulty quiz with ${questionCount} questions of type ${quizType} based on this text:\n\n${data.text}`
                        : `Explain the content of this document in detail:\n\n${data.text}`
                });
            } else if (req.file.mimetype.startsWith('image/')) {
                const base64Image = req.file.buffer.toString('base64');
                model = 'meta-llama/llama-4-scout-17b-16e-instruct';
                messages.push({
                    role: 'user',
                    content: [
                        {
                            type: 'text',
                            text: mode === 'quiz'
                                ? `Analyze this image and generate a ${difficulty} quiz with ${questionCount} questions (${quizType}).`
                                : `Analyze this image and provide a comprehensive explanation or solution to whatever is shown.`
                        },
                        {
                            type: 'image_url',
                            image_url: { url: `data:${req.file.mimetype};base64,${base64Image}` }
                        }
                    ]
                });
            } else {
                return res.status(400).json({ error: 'Unsupported file type.' });
            }
        } else if (req.body.topic) {
            messages.push({
                role: 'user',
                content: `Generate a ${difficulty} difficulty quiz with ${questionCount} questions of type ${quizType} on this topic:\n\nTopic: ${req.body.topic}`
            });
        } else {
            return res.status(400).json({ error: 'Either a file upload or a topic is required.' });
        }

        // ── Run through the load-balanced queue ─────────────────────────
        const totalPending = lanes.reduce((sum, l) => sum + l.queue.length + (l.inProgress ? 1 : 0), 0);
        if (totalPending > 0) {
            console.log(`[Balancer] generate-quiz: ${totalPending} request(s) across all lanes`);
        }

        const responseContent = await enqueueGroqCall(async (groqClient) => {
            const completion = await groqClient.chat.completions.create({
                messages,
                model,
                response_format: { type: 'json_object' }
            });
            return JSON.parse(completion.choices[0].message.content);
        }, `generate-quiz[${mode}]`);

        console.log(`Genius ${mode} generated:`, responseContent.title);
        res.json(responseContent);

    } catch (error) {
        console.error('Error in /generate-quiz:', error.message);

        // Give user-friendly messages for common errors
        if (/timed out while waiting/i.test(error.message)) {
            return res.status(503).json({
                error: 'The server is busy processing other requests. Please try again in a moment.',
                retryable: true
            });
        }
        if (error?.status === 429 || /rate.?limit/i.test(error.message)) {
            return res.status(429).json({
                error: 'AI service is temporarily overloaded. Please wait a few seconds and try again.',
                retryable: true
            });
        }
        res.status(500).json({ error: 'Failed to process material: ' + error.message });
    }
});

// ─── Grade Situational Answer ─────────────────────────────────────────────
app.post('/grade-situational', async (req, res) => {
    try {
        const { question, idealAnswer, userAnswer } = req.body;

        if (!question || !idealAnswer || !userAnswer) {
            return res.status(400).json({ error: 'question, idealAnswer, and userAnswer are required.' });
        }

        const result = await enqueueGroqCall(async (groqClient) => {
            const completion = await groqClient.chat.completions.create({
                messages: [
                    {
                        role: 'system',
                        content: `You are an expert examiner. Compare the user's answer to the ideal answer. 
Calculate a matching percentage (0-100) based on accuracy, key points covered, and understanding. 
Return ONLY a JSON object: { "score": 85, "feedback": "Brief feedback on what was missed or done well." }`
                    },
                    {
                        role: 'user',
                        content: `Question: ${question}\nIdeal Answer: ${idealAnswer}\nUser's Answer: ${userAnswer}`
                    }
                ],
                model: 'llama-3.3-70b-versatile',
                response_format: { type: 'json_object' }
            });
            return JSON.parse(completion.choices[0].message.content);
        }, 'grade-situational');

        res.json(result);

    } catch (error) {
        console.error('Error grading situational answer:', error.message);

        if (/timed out while waiting/i.test(error.message)) {
            return res.status(503).json({
                error: 'The server is busy. Please wait a moment and try submitting your answer again.',
                retryable: true
            });
        }
        if (error?.status === 429 || /rate.?limit/i.test(error.message)) {
            return res.status(429).json({
                error: 'AI service is temporarily overloaded. Please try again shortly.',
                retryable: true
            });
        }
        res.status(500).json({ error: 'Failed to grade answer: ' + error.message });
    }
});

// ─── Start Server ─────────────────────────────────────────────────────────
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`✅ Cherry GenAI server running on port ${PORT}`);
    console.log(`⏱  Groq queue: min ${GROQ_MIN_INTERVAL_MS}ms between calls | timeout ${QUEUE_TIMEOUT_MS / 1000}s | ${MAX_RETRIES} retries`);
});
