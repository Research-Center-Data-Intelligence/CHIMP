const datasetName = new URLSearchParams(window.location.search).get("dataset");
const carouselList = document.getElementById("carouselList");
const labeledData = [];

const socket = io(CONFIG.SOCKET_URL);
socket.on('connect', () => console.log('Initialised SocketIO connection...'));
socket.on('disconnect', () => console.log('Terminated SocketIO connection.'));

let imageData = [];
let splide = null;

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
      alert("Fout bij labelen: " + response.error);
    } else {
      console.log("Succesvol gelabeld:", response.filename);
      labeledData.push({
        filename: response.filename,
        emotion: response.emotion
      });

      const index = imageData.findIndex(item => item.filename === response.filename);
      if (index !== -1) {
        // Verwijder uit Splide en imageData
        splide.remove(index);
        imageData.splice(index, 1);

        if (splide.length === 0) {
          alert("Alle afbeeldingen zijn gelabeld!");
          document.querySelector("#imageCarousel").style.display = "none";
        } else {
          splide.go(">");
        }

        console.log(`Voortgang: ${labeledData.length} / ${labeledData.length + imageData.length}`);
      } else {
        console.warn("Kon afbeelding niet vinden in imageData");
      }
    }
  });
}

window.onload = async () => {
  await loadImages();
  setupButtons();
};
