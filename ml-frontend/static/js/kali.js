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
    const emotionOrderList = document.getElementById('emotionOrderList');

    let mediaRecorder;
    let currentEmotion = '';
    let recordedBlob;
    let isRecording = false;
    let isPaused = false;
    let emotionQueue = [];
    let recordedSessions = [];
    let isQueuePaused = false;
    const socket = io(CONFIG.SOCKET_URL);

    const recordedEmotions = new Set(); // Track recorded emotions

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
                markEmotionAsRecorded(currentEmotion); // Mark emotion as recorded
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

            // Reset recorded emotions
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
        if (recordedSessions.length === 0) {
            console.warn('No recorded data to save.');
            return;
        }

        const timestamp = new Date().toLocaleString('sv-SE', { timeZone: 'Europe/Amsterdam' }).replace(/[: ]/g, '-');
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
        
        processEmotionQueue();
    });

    emotionButtons.forEach(button => {
        button.addEventListener('click', () => {
            const emotion = button.getAttribute('data-emotion').toLowerCase();
             
            if (!isRecording) { 
                startRecording(emotion);
            } else {
                console.warn('Already recording. Please wait for the current recording to finish.');
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

    // Function to mark an emotion as recorded
    function markEmotionAsRecorded(emotion) {
        recordedEmotions.add(emotion.toLowerCase());
        // Find existing list item and add 'recorded' class
        const emotionListItem = document.querySelector(`#emotionOrderList li:nth-child(${Object.keys(CONFIG.EMOTION_SMILEYS).indexOf(emotion) + 1})`);
        if (emotionListItem) {
            emotionListItem.classList.add('recorded');
        }
    }

    // Function to reset recorded emotions
    function resetRecordedEmotions() {
        recordedEmotions.clear();
        // Remove 'recorded' class from all list items
        const listItems = emotionOrderList.getElementsByTagName('li');
        for (const listItem of listItems) {
            listItem.classList.remove('recorded');
        }
    }

    // Populate emotion list with initial state
    for (const emotion in CONFIG.EMOTION_SMILEYS) {
        const listItem = document.createElement('li');
        listItem.textContent = `${CONFIG.EMOTION_SMILEYS[emotion]} ${emotion.charAt(0).toUpperCase() + emotion.slice(1)}`;
        
        // Check if emotion is recorded
        if (recordedEmotions.has(emotion)) {
            listItem.classList.add('recorded'); 
        }

        emotionOrderList.appendChild(listItem);
    }
});
