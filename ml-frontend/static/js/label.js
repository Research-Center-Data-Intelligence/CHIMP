const datasetName = new URLSearchParams(window.location.search).get("dataset");
const carouselList = document.getElementById("carouselList");
const labeledData = [];

const socket = io(CONFIG.SOCKET_URL);
socket.on('connect', () => console.log('Initialised SocketIO connection...'));
socket.on('disconnect', () => console.log('Terminated SocketIO connection.'));

let imageData = [];
let splide = null;
let totalImages = 0;

async function loadImages() {
  console.log("Loading images for dataset:", datasetName);

  const res = await fetch(`/api/labeling_task_data/${datasetName}`);
  console.log("Fetched labeling task data, status:", res.status);

  if (!res.ok) {
    console.error("Fout bij ophalen labeling task data:", await res.text());
    return;
  }

  const json = await res.json();

  imageData = Object.entries(json.images).map(([filename, hexData]) => {
    const base64 = hexToBase64(hexData);
    return { filename, data: base64 };
  });

  totalImages = json.total_images;
  const initialLabeled = json.num_labeled;

  updateProgress(initialLabeled, totalImages);

  console.log("Loaded and converted image data:", imageData);

  imageData.forEach((item, index) => {
    const li = document.createElement("li");
    li.className = "splide__slide";
    li.innerHTML = `<img src="data:image/png;base64,${item.data}" data-filename="${item.filename}" data-index="${index}" alt="image" />`;
    carouselList.appendChild(li);
  });

  splide = new Splide("#imageCarousel", {
    perPage: 1,
    pagination: false,
    arrows: true,
  });

  splide.mount();
  console.log("Splide carousel mounted");
}

function hexToBase64(hex) {
  if (typeof hex !== 'string') {
    console.warn("hexToBase64: verwachte string maar kreeg:", typeof hex);
    return "";
  }

  const matches = hex.match(/[\da-f]{2}/gi);
  if (!matches) {
    console.warn("hexToBase64: geen geldige hex gevonden in:", hex);
    return "";
  }

  const bytes = new Uint8Array(matches.map(h => parseInt(h, 16)));
  let binary = "";
  bytes.forEach(b => binary += String.fromCharCode(b));
  return btoa(binary);
}

function base64ToBlob(base64, mime = "image/png") {
  const byteCharacters = atob(base64);
  const byteNumbers = new Array(byteCharacters.length);
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNumbers);
  return new Blob([byteArray], { type: mime });
}

function setupButtons() {
  const buttons = document.querySelectorAll(".emotionButton");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const currentSlide = document.querySelector(".splide__slide.is-active");
      const current = currentSlide?.querySelector("img");
      if (!current) return;

      const filename = current.getAttribute("data-filename");
      const emotion = btn.getAttribute("data-emotion");

      socket.emit("label_image", {
        dataset_name: datasetName,
        filename: filename,
        emotion: emotion
      });

      console.log(`Verzonden label voor ${filename}: ${emotion}`);
    });
  });

  socket.on("label_image_response", (response) => {
    if (response.error) {
      alert("Error labeling image: " + response.error);
    } else {
      console.log("Successfully labeled:", response.filename);
      labeledData.push({
        filename: response.filename,
        emotion: response.emotion
      });

      const index = imageData.findIndex(item => item.filename === response.filename);
      if (index !== -1) {
        splide.remove(index);
        imageData.splice(index, 1);

        const currentLabeled = labeledData.length;
        updateProgress(currentLabeled, totalImages);

        if (imageData.length === 0) {
          setTimeout(() => {
            window.location.href = "/unlabeled";  
          }, 500); 
        } else {
          splide.go(">");
        }

        console.log(`Progress: ${currentLabeled} / ${totalImages}`);
      } else {
        console.warn("Could not find image in imageData");
      }
    }
  });
}

function updateProgress(labeledCount, total) {
  const percent = total === 0 ? 0 : Math.round((labeledCount / total) * 100);
  document.getElementById("labeledCount").textContent = labeledCount;
  document.getElementById("totalCount").textContent = total;
  document.getElementById("labelProgress").style.width = percent + "%";
}

window.onload = async () => {
  await loadImages();
  setupButtons();
};
