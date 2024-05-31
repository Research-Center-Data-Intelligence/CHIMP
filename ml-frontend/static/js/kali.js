document.addEventListener('DOMContentLoaded', () => {
    const videoElement = document.getElementById('videoElement');
    const countdownOverlay = document.getElementById('countdownOverlay');
    const countdownText = document.getElementById('countdownText');
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const pauseButton = document.getElementById('pauseButton');
    const resumeButton = document.getElementById('resumeButton');
    const saveButton = document.getElementById('saveButton');
    const emotionButtons = document.querySelectorAll('.emotionButton');
    

    let mediaRecorder;
    let currentEmotion = '';
    let recordedBlob;
    let isRecording = false;
    let isPaused = false;
    let emotionQueue = [];
    let recordedSessions = [];
    let isQueuePaused = false;
    const socket = io(CONFIG.SOCKET_URL);

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
            countdownOverlay.style.display = 'flex';
            const smiley = CONFIG.EMOTION_SMILEYS[emotion] || '';
            countdownText.textContent = `${CONFIG.MAX_RECORDING_TIME} - ${smiley} ${emotion}`;

            let countdown = CONFIG.MAX_RECORDING_TIME;
            const countdownInterval = setInterval(() => {
                countdown--;
                countdownText.textContent = `${countdown} - ${smiley} ${emotion}`;
                if (countdown === 0) {
                    clearInterval(countdownInterval);
                    countdownOverlay.style.display = 'none';
                    resolve();
                }
            }, 1000);
        });
    }

    async function startRecording(emotion = '') {
        if (isRecording) {
            console.warn('Already recording.');
            return;
        }

        currentEmotion = emotion;
        await startRecordingWithCountdown(currentEmotion);  

        const options = { mimeType: 'video/x-matroska;codecs=avc1' };
        mediaRecorder = new MediaRecorder(videoElement.srcObject, options);
        recordedBlob = null;

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                recordedBlob = event.data;
            }
        };

        mediaRecorder.onstop = () => {
            console.log(`Recording ${currentEmotion ? 'for ' + currentEmotion : ''} stopped.`);
            if (recordedBlob) {
                recordedSessions.push({ emotion: currentEmotion, blob: recordedBlob });
                saveButton.disabled = false;
            }
            isRecording = false;
            if (!isQueuePaused) {
                processEmotionQueue();
            }
        };

        mediaRecorder.start();
        console.log(`Recording ${emotion ? 'for ' + emotion : ''} started...`);
        isRecording = true;
        isPaused = false;

        setTimeout(() => {
            stopRecording();
        }, CONFIG.MAX_RECORDING_TIME * 1000);

        updateButtonState();
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.requestData();
            mediaRecorder.stop();
            isRecording = false;
            clearHighlightEmotionButton();
            updateButtonState();
        }
    }

    function pauseRecordingQueue() {
        if (isRecording) {
            stopRecording();
        }
        isQueuePaused = true;
        console.log('Recording paused...');
        updateButtonState();
    }

    function resumeRecordingQueue() {
        isQueuePaused = false;
        console.log('Recording resumed...');
        processEmotionQueue();
        updateButtonState();
    }

    function saveRecording() {
        if (recordedSessions.length === 0) {
            console.warn('No recorded data to save.');
            return;
        }

        const timestamp = new Date().toLocaleString('sv-SE', { timeZone: 'Europe/Amsterdam' }).replace(/[: ]/g, '-');
        //const username = usernameInput.value || 'anonymous';
        const username = USERNAME;

        recordedSessions.forEach((session, index) => {
            const blob = session.blob;
            const emotion = session.emotion;

            socket.emit('process-video', {
                user_id: '', 
                username: username,
                image_blob: blob,
                emotion: emotion,
                timestamp: `${timestamp}-${index}`
            });
            console.log(`Recording for ${emotion} saved.`);
        });

        recordedSessions = []; 
        saveButton.disabled = true;
        updateButtonState();
    }

    function updateButtonState() {
        startButton.disabled = isRecording || emotionQueue.length > 0;
        stopButton.disabled = !isRecording && emotionQueue.length === 0;
        pauseButton.disabled = !isRecording || isPaused || emotionQueue.length === 0;
        resumeButton.disabled = !isQueuePaused || emotionQueue.length === 0;
        emotionButtons.forEach(button => button.disabled = isRecording);
    }

    function processEmotionQueue() {
        if (emotionQueue.length > 0 && !isQueuePaused) {
            const nextEmotion = emotionQueue.shift();
            startRecording(nextEmotion);
        }
    }

    function highlightEmotionButton(emotion) {
        clearHighlightEmotionButton();
        const button = document.querySelector(`.emotionButton[data-emotion="${emotion}"]`);
        if (button) {
            button.classList.add('highlight');
        }
    }

    function clearHighlightEmotionButton() {
        const highlightedButton = document.querySelector('.emotionButton.highlight');
        if (highlightedButton) {
            highlightedButton.classList.remove('highlight');
        }
    }

    function resetRecordingState() {
        isRecording = false;
        isPaused = false;
        isQueuePaused = false;
        emotionQueue = [];
        clearHighlightEmotionButton();
        updateButtonState();
    }

    emotionButtons.forEach(button => {
        button.addEventListener('click', () => {
            const emotion = button.getAttribute('data-emotion');
            emotionQueue.push(emotion);
            processEmotionQueue();
        });
    });

    startButton.addEventListener('click', () => {
        emotionButtons.forEach(button => {
            const emotion = button.getAttribute('data-emotion');
            emotionQueue.push(emotion);
        });
        processEmotionQueue();
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
});
