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
    const socket = io('http://localhost:5252'); // Connect to your Flask-SocketIO server

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

    async function startRecordingWithCountdown() {
        return new Promise((resolve) => {
            countdownOverlay.style.display = 'flex';
            let countdown = 3;
            countdownText.textContent = countdown;

            const countdownInterval = setInterval(() => {
                countdown--;
                countdownText.textContent = countdown;
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
        await startRecordingWithCountdown();  

        var options = {mimeType: 'video/x-matroska;codecs=avc1'};
        mediaRecorder = new MediaRecorder(videoElement.srcObject,options);
        recordedBlob = null;

        mediaRecorder.ondataavailable = (event) => {
            console.log(event);
            if (event.data.size > 0) {
                recordedBlob = event.data;
            }
        };

        mediaRecorder.onstop = () => {
            console.log(`Recording ${currentEmotion ? 'for ' + currentEmotion : ''} stopped.`);
            if (recordedBlob) {
                saveButton.disabled = false;
            }
        };

        mediaRecorder.start();
        console.log(`Recording ${emotion ? 'for ' + emotion : ''} started...`);
        isRecording = true;
        isPaused = false;
        updateButtonState();
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.requestData();
            mediaRecorder.stop();
            isRecording = false;
            updateButtonState();
        }
    }

    function pauseRecording() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.pause();
            console.log('Recording paused...');
            isPaused = true;
            updateButtonState();
        }
    }

    function resumeRecording() {
        if (mediaRecorder && mediaRecorder.state === 'paused') {
            mediaRecorder.resume();
            console.log('Recording resumed...');
            isPaused = false;
            updateButtonState();
        }
    }

    function saveRecording() {
        console.log(recordedBlob);
        
        if (!recordedBlob) {
            console.warn('No recorded data to save.');
            return;
        }
    
        
        var blob = recordedBlob;
const timestamp = new Date().toLocaleString('sv-SE', { timeZone: 'Europe/Amsterdam' }).replace(/[: ]/g, '-');
        socket.emit('process-video', {
            
            user_id: '', 
            image_blob: blob,
            emotion: currentEmotion,
            timestamp: timestamp
        });
        console.log('gestuurd')
        console.log(`Recording ${currentEmotion ? 'for ' + currentEmotion : ''} saved.`);
        recordedBlob = null;
        saveButton.disabled = true;
        updateButtonState();
    }
    

    function updateButtonState() {
        startButton.disabled = isRecording;
        stopButton.disabled = !isRecording;
        pauseButton.disabled = !isRecording || isPaused;
        resumeButton.disabled = !isRecording || !isPaused;
        emotionButtons.forEach(button => button.disabled = isRecording);
    }

    startButton.addEventListener('click', () => startRecording());
    stopButton.addEventListener('click', stopRecording);
    pauseButton.addEventListener('click', pauseRecording);
    resumeButton.addEventListener('click', resumeRecording);
    saveButton.addEventListener('click', saveRecording);

    emotionButtons.forEach(button => {
        button.addEventListener('click', () => startRecording(button.getAttribute('data-emotion')));
    });

    setupCamera().catch(error => {
        console.error('Failed to initialize webcam:', error);
    });
});
