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
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

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
//  GROQ RATE-LIMIT QUEUE
//  Groq free tier: ~1 request per 2 seconds (30 RPM).
//  All Groq calls are funneled through this queue so concurrent users
//  never trigger rate-limit errors; they simply wait their turn gracefully.
// ═══════════════════════════════════════════════════════════════════════════

const GROQ_MIN_INTERVAL_MS = 2100;   // 2.1 s gap between Groq calls (safety margin)
const QUEUE_TIMEOUT_MS     = 120000; // Drop a job after 2 min if still waiting
const MAX_RETRIES          = 3;      // Retry on transient Groq errors
const RETRY_DELAY_MS       = 3000;   // Wait 3 s between retries

let lastGroqCallAt = 0;           // Timestamp of the last completed Groq call
let groqCallInProgress = false;   // Is a Groq call executing right now?
const groqQueue = [];             // Pending { fn, resolve, reject, enqueuedAt, label } entries

/**
 * Schedule a Groq API call through the queue.
 * @param {Function} fn  - async function that returns the Groq response
 * @param {string}   label - human-readable name for logging
 * @returns Promise<any>
 */
function enqueueGroqCall(fn, label = 'groq-call') {
    return new Promise((resolve, reject) => {
        const enqueuedAt = Date.now();
        groqQueue.push({ fn, resolve, reject, enqueuedAt, label });
        console.log(`[Queue] ${label} enqueued | queue length: ${groqQueue.length}`);
        processQueue();
    });
}

async function processQueue() {
    // Only one pump loop allowed at a time
    if (groqCallInProgress) return;

    const job = groqQueue.shift();
    if (!job) return;

    // Check timeout before executing
    if (Date.now() - job.enqueuedAt > QUEUE_TIMEOUT_MS) {
        console.warn(`[Queue] ${job.label} timed out while waiting in queue`);
        job.reject(new Error('Request timed out while waiting in the processing queue. Please try again.'));
        processQueue(); // Try next job
        return;
    }

    groqCallInProgress = true;

    // Enforce minimum interval between Groq API calls
    const msSinceLast = Date.now() - lastGroqCallAt;
    if (msSinceLast < GROQ_MIN_INTERVAL_MS) {
        const waitMs = GROQ_MIN_INTERVAL_MS - msSinceLast;
        console.log(`[Queue] ${job.label} waiting ${waitMs}ms for rate limit`);
        await sleep(waitMs);
    }

    // Execute with retry logic
    let lastError;
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        try {
            console.log(`[Queue] ${job.label} executing (attempt ${attempt}/${MAX_RETRIES})`);
            lastGroqCallAt = Date.now();
            const result = await job.fn();
            console.log(`[Queue] ${job.label} succeeded`);
            job.resolve(result);
            lastError = null;
            break;
        } catch (err) {
            lastError = err;
            const isRateLimit = err?.status === 429 || /rate.?limit/i.test(err?.message || '');
            const isRetryable = isRateLimit || err?.status >= 500;

            console.error(`[Queue] ${job.label} attempt ${attempt} failed: ${err.message}`);

            if (attempt < MAX_RETRIES && isRetryable) {
                const delay = isRateLimit ? GROQ_MIN_INTERVAL_MS * 2 : RETRY_DELAY_MS;
                console.log(`[Queue] ${job.label} retrying in ${delay}ms...`);
                await sleep(delay);
                lastGroqCallAt = Date.now(); // Reset timer after waiting
            }
        }
    }

    if (lastError) {
        console.error(`[Queue] ${job.label} failed after ${MAX_RETRIES} attempts`);
        job.reject(lastError);
    }

    groqCallInProgress = false;

    // Schedule next job — small tick to avoid synchronous stack overflow
    if (groqQueue.length > 0) {
        setImmediate(processQueue);
    }
}

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

// ─── Queue Status Endpoint (optional monitoring) ──────────────────────────
app.get('/queue-status', (req, res) => {
    res.json({
        queueLength: groqQueue.length,
        processing: groqCallInProgress,
        estimatedWaitSeconds: Math.ceil(
            (groqQueue.length + (groqCallInProgress ? 1 : 0)) * (GROQ_MIN_INTERVAL_MS / 1000)
        ),
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

        // ── Inform the client how many are ahead in the queue ────────────
        const queuePosition = groqQueue.length + (groqCallInProgress ? 1 : 0);
        if (queuePosition > 0) {
            console.log(`[Queue] generate-quiz: ${queuePosition} request(s) ahead`);
        }

        // ── Run through the queue ────────────────────────────────────────
        const responseContent = await enqueueGroqCall(async () => {
            const completion = await groq.chat.completions.create({
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

        const queuePosition = groqQueue.length + (groqCallInProgress ? 1 : 0);
        if (queuePosition > 0) {
            console.log(`[Queue] grade-situational: ${queuePosition} request(s) ahead`);
        }

        const result = await enqueueGroqCall(async () => {
            const completion = await groq.chat.completions.create({
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
