document.addEventListener('DOMContentLoaded', () => {
    const videoElement = document.getElementById('videoElement');
    const countdownOverlay = document.getElementById('countdownOverlay');
    const countdownText = document.getElementById('countdownText');
    const startButton = document.getElementById('startButton');
    const poolButton = document.getElementById('poolButton');
    const stopButton = document.getElementById('stopButton');
    const pauseButton = document.getElementById('pauseButton');
    const resumeButton = document.getElementById('resumeButton');
    const saveButton = document.getElementById('saveButton');
    const emotionButtons = document.querySelectorAll('.emotionButton');
    const emotionOrderList = document.getElementById('emotionOrderList');

    let mediaRecorder;
    let currentEmotion = '';
    let recordedBlob;
    let isRecording = false;
    let isPaused = false;
    let emotionQueue = [];
    let recordedSessions = [];
    let isQueuePaused = false;
    let chunks = [];

    const socket = io(CONFIG.SOCKET_URL);

    socket.on('connect', () => console.log('Initialised SocketIO connection...'));
    socket.on('disconnect', () => console.log('Terminated SocketIO connection.'));

    const recordedEmotions = new Set();

    async function setupCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            videoElement.srcObject = stream;
            await new Promise((resolve) => {
                videoElement.onloadedmetadata = () => {
                    videoElement.play();
                    resolve();
                };
            });
        } catch (error) {
            console.error('Error accessing webcam:', error);
            alert('Failed to access webcam.');
        }
    }

        async function startRecordingWithCountdown(emotion) {
        return new Promise((resolve) => {
            countdownOverlay.style.display = 'flex'; // Show countdown overlay
            const smiley = CONFIG.EMOTION_SMILEYS[emotion] || ''; // Get smiley for the emotion
            countdownText.textContent = `${CONFIG.COUNTDOWN_TIME} - ${smiley} ${emotion}`; // Set countdown text

            let countdown = CONFIG.COUNTDOWN_TIME; // Initialize countdown time
            const countdownInterval = setInterval(() => {
                countdown--;
                countdownText.textContent = `${countdown} - ${smiley} ${emotion}`; // Update countdown text
                if (countdown === 0) {
                    clearInterval(countdownInterval);
                    countdownOverlay.style.display = 'none'; // Hide countdown overlay
                    resolve(); // Resolve the promise after countdown ends
                }
            }, 1000);
        });
    }


    async function startSegmentedPoolRecording() {
        const segmentDuration = CONFIG.MAX_RECORDING_TIME; // Set segment duration
        // Calculate the total number of segments based on the maximum pool recording time;
        const totalSegments = CONFIG.MAX_POOL_RECORDING_TIME / segmentDuration;

        for (let i = 0; i < totalSegments; i++) {
            console.log(`[INFO] Segment ${i + 1}/${totalSegments} starting`);
            await startRecording('', true, segmentDuration);
            await wait(segmentDuration * 1000 + 500); 
        }

        console.log('[INFO] Finished segmented pool recording. Saving...');
    }


    // Function to start recording a video segment
    async function startRecording(emotion = '', isPool = false, durationOverride = null) {
        // Prevent starting a new recording if one is already in progress
        if (isRecording) {
            console.warn('Already recording.');
            return;
        }

        // Set the current emotion being recorded
        currentEmotion = emotion;
        console.log('[DEBUG] startRecording triggered for emotion:', emotion);

        // Start the countdown before recording begins
        await startRecordingWithCountdown(currentEmotion);

        // Configure the MediaRecorder with video settings
        const options = { mimeType: 'video/webm;codecs=vp9' };
        mediaRecorder = new MediaRecorder(videoElement.srcObject, options);
        recordedBlob = null; // Reset the recorded blob
        chunks = []; // Clear any previous chunks

        // Event handler for when data is available from the MediaRecorder
        mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                chunks.push(event.data); // Collect video data chunks
            }
        };

        // Event handler for when the MediaRecorder stops
        mediaRecorder.onstop = () => {
            // Combine all chunks into a single Blob
            recordedBlob = new Blob(chunks, { type: 'video/webm' });
            if (recordedBlob) {
                // Determine if the recording is unlabeled
                const isUnlabeled = !currentEmotion || currentEmotion === 'unlabeled';
                // Create a session object to store recording details
                const session = {
                    emotion: isUnlabeled ? 'unlabeled' : currentEmotion,
                    blob: recordedBlob,
                    isUnlabeled: isUnlabeled,
                    isPool: isPool
                };
                // Add the session to the list of recorded sessions
                recordedSessions.push(session);
                saveButton.disabled = false; // Enable the save button
                // Mark the emotion as recorded if it's not unlabeled
                if (!isUnlabeled) {
                    markEmotionAsRecorded(currentEmotion);
                }
            }

            // Reset recording state
            isRecording = false;
            countdownOverlay.style.display = 'none'; // Hide the countdown overlay

            // Process the next emotion in the queue if the queue is not paused
            if (!isQueuePaused) {
                processEmotionQueue();
            }
        };

        // Start the MediaRecorder
        mediaRecorder.start();
        isRecording = true; // Set recording state to true
        isPaused = false; // Ensure paused state is false

        // Determine the recording duration
        const duration = durationOverride ?? (isPool ? CONFIG.MAX_POOL_RECORDING_TIME : CONFIG.MAX_RECORDING_TIME);

        // Automatically stop recording after the specified duration
        setTimeout(() => {
            stopRecording();
        }, duration * 1000);

        // Update the state of buttons
        updateButtonState();
    }

    // Helper function to wait for a specified duration
    function wait(ms) {
        return new Promise(resolve => setTimeout(resolve, ms)); // return a promise that resolves after the specified duration
    }



    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.requestData();
            mediaRecorder.stop();
            isRecording = false;
            resetRecordedEmotions();
            updateButtonState(); 
        }
    }

    function pauseRecordingQueue() {
        if (isRecording) {
            stopRecording();
        }
        isQueuePaused = true;
        console.log('Recording queue paused...');
        updateButtonState();
    }

    function resumeRecordingQueue() {
        isQueuePaused = false;
        console.log('Recording queue resumed...');
        processEmotionQueue();
        updateButtonState();
    }

    function saveRecording() {
        console.log('[DEBUG] saveRecording triggered');
        console.log('[DEBUG] recordedSessions:', recordedSessions);

        if (recordedSessions.length === 0) {
            alert('No recorded data available to save.');
            return;
        }

        const timestamp = new Date()
            .toLocaleString('sv-SE', { timeZone: 'Europe/Amsterdam' })
            .replace(/[: ]/g, '-');
        const username = USERNAME;

        const blobs = recordedSessions.map(session => session.blob);
        const emotions = recordedSessions.map(session =>
            session.isUnlabeled ? 'unlabeled' : session.emotion
        );
        const timestamps = recordedSessions.map((_, index) => `${timestamp}-${index}`);
        const isPool = recordedSessions.some(session => session.isPool);

        console.log(`[DEBUG] Total sessions to upload: ${recordedSessions.length}`);
        recordedSessions.forEach((session, i) => {
            console.log(`[DEBUG] Session ${i}: Emotion = ${session.emotion}, isPool = ${session.isPool}, Blob size = ${session.blob?.size ?? 'N/A'} bytes`);
        });

        const payload = {
            user_id: '',
            username: username,
            image_blobs: blobs,
            emotions: emotions,
            timestamps: timestamps,
            ...(isPool && { is_pool: true })
        };

        console.log('[DEBUG] Payload prepared:', {
            blobCount: blobs.length,
            emotions,
            timestamps,
            totalBlobSize: blobs.reduce((acc, b) => acc + (b?.size ?? 0), 0)
        });


        if (isPool) {
            socket.emit('upload_managed_pool_data', payload);
            console.log('Pool recording saved.');
        } else {
            socket.emit('upload_managed_calibration_data', payload);
            console.log('Calibration recording saved.');
        }

        recordedSessions = [];
        saveButton.disabled = true;
        updateButtonState();
    }

    function updateButtonState() {
        startButton.disabled = isRecording || emotionQueue.length > 0;
        poolButton.disabled = isRecording;
        stopButton.disabled = !isRecording && emotionQueue.length === 0;
        pauseButton.disabled = !isRecording || isPaused || emotionQueue.length === 0;
        resumeButton.disabled = !isQueuePaused || emotionQueue.length === 0;
        emotionButtons.forEach(button => button.disabled = isRecording);
    }

    function processEmotionQueue() {
        if (emotionQueue.length > 0 && !isQueuePaused) {
            const nextEmotion = emotionQueue.shift();
            console.log('[DEBUG] Processing next emotion in queue:', nextEmotion);
            startRecording(nextEmotion);
        }
    }

    function resetRecordingState() {
        isRecording = false;
        isPaused = false;
        isQueuePaused = false;
        emotionQueue = [];
        updateButtonState();
    }

    startButton.addEventListener('click', () => {
        for (const emotion in CONFIG.EMOTION_SMILEYS) {
            emotionQueue.push(emotion);
        }
        console.log('[DEBUG] Emotion queue initialized:', emotionQueue);
        processEmotionQueue();
    });

    poolButton.addEventListener('click', async () => {
        if (isRecording) {
            console.warn('Already recording. Please wait.');
            return;
        }
        console.log('Starting segmented recording for unlabeled pool data...');
        await startSegmentedPoolRecording();
    });



    emotionButtons.forEach(button => {
        button.addEventListener('click', () => {
            const emotion = button.getAttribute('data-emotion').toLowerCase();
            console.log('[DEBUG] Emotion button clicked:', emotion);
            if (!isRecording) {
                startRecording(emotion);
            } else {
                console.warn('Already recording.');
            }
        });
    });

    stopButton.addEventListener('click', () => {
        isQueuePaused = true;
        emotionQueue = [];
        stopRecording();
        resetRecordingState();
    });

    pauseButton.addEventListener('click', pauseRecordingQueue);
    resumeButton.addEventListener('click', resumeRecordingQueue);
    saveButton.addEventListener('click', saveRecording);

    setupCamera().catch(error => {
        console.error('Failed to initialize webcam:', error);
    });

    function markEmotionAsRecorded(emotion) {
        recordedEmotions.add(emotion.toLowerCase());
        const emotionListItem = document.querySelector(`#emotionOrderList li:nth-child(${Object.keys(CONFIG.EMOTION_SMILEYS).indexOf(emotion) + 1})`);
        if (emotionListItem) {
            emotionListItem.classList.add('recorded');
        }
    }

    function resetRecordedEmotions() {
        recordedEmotions.clear();
        const listItems = emotionOrderList.getElementsByTagName('li');
        for (const listItem of listItems) {
            listItem.classList.remove('recorded');
        }
    }

    for (const emotion in CONFIG.EMOTION_SMILEYS) {
        const listItem = document.createElement('li');
        listItem.textContent = `${CONFIG.EMOTION_SMILEYS[emotion]} ${emotion.charAt(0).toUpperCase() + emotion.slice(1)}`;
        if (recordedEmotions.has(emotion)) {
            listItem.classList.add('recorded');
        }
        emotionOrderList.appendChild(listItem);
    }
});