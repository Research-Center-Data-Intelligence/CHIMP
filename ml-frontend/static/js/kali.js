async function setupCamera() {
    const videoElement = document.getElementById('videoElement');

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;

        // wacht op video beeld en meta data
        await new Promise((resolve) => {
            videoElement.onloadedmetadata = () => {
                console.log('Video metadata loaded');
                videoElement.play();
                resolve();
            };
        });
    } catch (error) {
        console.error('Error accessing webcam:', error);
        alert('Failed to access webcam.');
    }
}

async function takeSnapshot() {
    const videoElement = document.getElementById('videoElement');
    const canvasElement = document.getElementById('canvasElement');
    const canvasContext = canvasElement.getContext('2d');

    try {
        
        await setupCamera();

        // Wacht dat beeld laat
        await new Promise(resolve => videoElement.onloadeddata = resolve);

        // canvas bepalen voor beeld webcam
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;

        // beeld laten zien
        canvasContext.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

        
        const imageData = canvasElement.toDataURL('image/png');

        
        const downloadLink = document.createElement('a');
        downloadLink.href = imageData;
        downloadLink.download = 'snapshot.png';
        downloadLink.click();

    } catch (error) {
        console.error('Error capturing snapshot:', error);
        alert('Failed to capture snapshot.');
    }
}


document.addEventListener('DOMContentLoaded', () => {
    const snapshotButton = document.getElementById('snapshotButton');
    snapshotButton.addEventListener('click', takeSnapshot);

    
    setupCamera().catch(err => console.error('Camera setup error:', err));
});
